"""Resilient Request Dispatcher coordinating rate limits, queueing, retries, and timeouts."""

import time
import uuid
import datetime
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from the_relay.models import (
    RequestStatus,
    SourceType,
    AnswerGrade,
    ResultsCsvRow,
)
from the_relay.pipeline.core import RelayPipeline
from the_relay.pipeline.models import PipelineResult, PipelineBranch, LatencyBreakdown
from the_relay.resilience.limiter import AdaptiveRateLimiter, RateLimiterPolicy
from the_relay.resilience.retry import RetryManager, RetryConfig


DAILY_BUDGET_EXHAUSTED_MESSAGE = (
    "The daily computational research quota has been exhausted. "
    "All available model credits have been utilized. Service will resume at 00:00 UTC."
)


class ResilientDispatcher:
    """Dispatches queries to the 4-step pipeline under strict concurrency and rate limit controls."""

    def __init__(
        self,
        pipeline: Optional[RelayPipeline] = None,
        limiter: Optional[AdaptiveRateLimiter] = None,
        retry_mgr: Optional[RetryManager] = None,
    ):
        self.pipeline = pipeline or RelayPipeline()
        self.limiter = limiter or AdaptiveRateLimiter()
        self.retry_mgr = retry_mgr or RetryManager()

    def dispatch(
        self,
        question: str,
        scenario_name: str = "default",
        request_id: Optional[str] = None,
    ) -> PipelineResult:
        """Dispatches a single question with queuing, rate limiting, and retries."""
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        arrival_time = time.perf_counter()
        start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.retry_mgr.record_attempt()

        # 1. Enqueue request
        in_queue, depth = self.limiter.enter_queue()
        if not in_queue:
            # Shed load: Queue full
            duration_ms = (time.perf_counter() - arrival_time) * 1000
            return self._build_shed_result(
                req_id=req_id,
                question=question,
                reason="Queue capacity exceeded (backpressure 429)",
                status=RequestStatus.RATE_LIMITED,
                duration_ms=duration_ms,
            )

        attempt = 0
        total_retries = 0
        total_retry_wait_ms = 0.0

        try:
            while True:
                # 2. Acquire execution slot from rate limiter
                slot_wait_start = time.perf_counter()
                slot_acquired = False
                wait_reason = None

                while not slot_acquired:
                    elapsed = time.perf_counter() - arrival_time
                    if elapsed >= self.limiter.policy.client_timeout_s:
                        duration_ms = elapsed * 1000
                        return self._build_shed_result(
                            req_id=req_id,
                            question=question,
                            reason="Client timed out waiting in queue",
                            status=RequestStatus.TIMEOUT,
                            duration_ms=duration_ms,
                        )

                    can_run, reason, wait_s = self.limiter.acquire_slot(now=time.time())
                    if can_run:
                        slot_acquired = True
                        break

                    # Check daily budget exhaustion (immediate shedding)
                    if reason == "DAILY_BUDGET_EXHAUSTED":
                        duration_ms = (time.perf_counter() - arrival_time) * 1000
                        return self._build_daily_exhausted_result(
                            req_id=req_id,
                            question=question,
                            duration_ms=duration_ms,
                        )

                    # Wait before next attempt to acquire slot
                    sleep_time = min(wait_s or 0.2, 0.5)
                    time.sleep(sleep_time)

                queue_wait_ms = (time.perf_counter() - slot_wait_start) * 1000

                # 3. Execute pipeline through slot
                try:
                    result = self.pipeline.process_query(
                        question=question,
                        request_id=req_id,
                        queue_wait_ms=queue_wait_ms,
                    )
                finally:
                    self.limiter.release_slot()

                # 4. Check if result requires retry
                if result.status in (RequestStatus.RATE_LIMITED, RequestStatus.ERROR):
                    elapsed_so_far = time.perf_counter() - arrival_time
                    can_retry, retry_msg = self.retry_mgr.can_retry(
                        attempt_number=attempt,
                        status_code_or_reason=result.status.value,
                        elapsed_request_s=elapsed_so_far,
                        client_timeout_s=self.limiter.policy.client_timeout_s,
                    )
                    if can_retry:
                        attempt += 1
                        total_retries += 1
                        backoff = self.retry_mgr.calculate_backoff(attempt)
                        total_retry_wait_ms += (backoff * 1000)
                        time.sleep(backoff)
                        continue  # Retry execution loop

                # Successful or non-retryable terminal response
                result.retries = total_retries
                result.latency.retrying_ms = total_retry_wait_ms
                result.latency.total_ms = (time.perf_counter() - arrival_time) * 1000
                return result

        finally:
            self.limiter.leave_queue()

    def _build_shed_result(
        self,
        req_id: str,
        question: str,
        reason: str,
        status: RequestStatus,
        duration_ms: float,
    ) -> PipelineResult:
        """Constructs an immediate fast rejection result when dropped or timed out."""
        return PipelineResult(
            request_id=req_id,
            question=question,
            answer=f"Request dropped: {reason}",
            citations=[],
            branch_taken=PipelineBranch.REFUSAL,
            loop_count=0,
            steps_executed=[],
            tool_calls=[],
            latency=LatencyBreakdown(
                waiting_ms=duration_ms,
                working_ms=0.0,
                retrying_ms=0.0,
                first_token_ms=0.0,
                total_ms=duration_ms,
                delay_before_first_token=True,
            ),
            status=status,
            source=self.pipeline.client.source,
            retries=0,
            tokens_in=len(question.split()) * 2,
            tokens_out=0,
            retrieval_setting_name=self.pipeline.engine.setting.display_name(),
        )

    def _build_daily_exhausted_result(
        self,
        req_id: str,
        question: str,
        duration_ms: float,
    ) -> PipelineResult:
        """Constructs the exact reply a user receives when the daily budget is gone (Section 7)."""
        return PipelineResult(
            request_id=req_id,
            question=question,
            answer=DAILY_BUDGET_EXHAUSTED_MESSAGE,
            citations=[],
            branch_taken=PipelineBranch.REFUSAL,
            loop_count=0,
            steps_executed=[],
            tool_calls=[],
            latency=LatencyBreakdown(
                waiting_ms=duration_ms,
                working_ms=1.0,
                retrying_ms=0.0,
                first_token_ms=duration_ms,
                total_ms=duration_ms + 1.0,
                delay_before_first_token=True,
            ),
            status=RequestStatus.RATE_LIMITED,
            source=self.pipeline.client.source,
            retries=0,
            tokens_in=len(question.split()) * 2,
            tokens_out=len(DAILY_BUDGET_EXHAUSTED_MESSAGE.split()),
            retrieval_setting_name=self.pipeline.engine.setting.display_name(),
        )

