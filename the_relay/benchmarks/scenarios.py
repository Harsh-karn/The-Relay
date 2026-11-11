"""Load test scenario generators for The Relay.

Implements the 5 mandatory scenarios:
1. Steady: Consistent, well-spaced requests within 30 RPM capacity
2. Burst: Sudden concurrent surge testing in-flight concurrency gates & queue buffering
3. Sustained Overload: Continuous high-frequency load exceeding 30 RPM, testing backpressure shedding
4. Per-minute limit halved: Artificially halved limits (15 RPM), testing degraded throughput
5. Daily budget exhausted: 3:00 PM condition where daily token quota is zero, testing graceful refusal
"""

import time
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from the_relay.models import (
    ResultsCsvRow,
    RequestStatus,
    SourceType,
    AnswerGrade,
)
from the_relay.resilience.dispatcher import ResilientDispatcher
from the_relay.retrieval.test_suite import TestSuiteManager, BenchmarkQuestion


class ScenarioResult:
    """Aggregated output of a scenario execution."""
    def __init__(self, scenario_name: str, source: SourceType):
        self.scenario_name = scenario_name
        self.source = source
        self.rows: List[ResultsCsvRow] = []
        self.duration_s: float = 0.0
        self.requests_sent: int = 0
        self.offered_rate_per_min: float = 0.0
        self.peak_queue_length: int = 0


class BenchmarkScenarios:
    """Drives the 5 mandatory benchmark load shapes."""

    def __init__(
        self,
        dispatcher: ResilientDispatcher,
        test_suite: Optional[TestSuiteManager] = None,
    ):
        self.dispatcher = dispatcher
        self.test_suite = test_suite or TestSuiteManager()
        self.questions = self.test_suite.load()

    def run_steady(
        self,
        n_requests: int = 15,
        interval_s: float = 2.0,
        source: SourceType = SourceType.STAND_IN,
    ) -> ScenarioResult:
        """Scenario 1: Steady load (offered rate ~30 req/min)."""
        self.dispatcher.limiter.reset()
        res = ScenarioResult("steady", source)
        start_time = time.perf_counter()

        for i in range(n_requests):
            q = self.questions[i % len(self.questions)]
            req_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            p_res = self.dispatcher.dispatch(q.question, scenario_name="steady")
            grade = self.test_suite.grade_answer(q, p_res.answer, p_res.citations)

            row = ResultsCsvRow(
                request_id=p_res.request_id,
                scenario="steady",
                source=source,
                start_time=req_start_iso,
                first_token_ms=p_res.latency.first_token_ms,
                total_ms=p_res.latency.total_ms,
                status=p_res.status,
                retries=p_res.retries,
                tokens_in=p_res.tokens_in,
                tokens_out=p_res.tokens_out,
                retrieval_setting=p_res.retrieval_setting_name,
                answer_grade=grade,
            )
            res.rows.append(row)
            time.sleep(interval_s)

        res.duration_s = max(1.0, time.perf_counter() - start_time)
        res.requests_sent = n_requests
        res.offered_rate_per_min = round((n_requests / res.duration_s) * 60, 1)
        res.peak_queue_length = self.dispatcher.limiter.peak_queue_depth
        return res

    def run_burst(
        self,
        burst_size: int = 20,
        source: SourceType = SourceType.STAND_IN,
    ) -> ScenarioResult:
        """Scenario 2: Sudden concurrent burst of requests arriving nearly simultaneously."""
        self.dispatcher.limiter.reset()
        res = ScenarioResult("burst", source)
        start_time = time.perf_counter()

        def worker(idx: int):
            q = self.questions[idx % len(self.questions)]
            req_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p_res = self.dispatcher.dispatch(q.question, scenario_name="burst")
            grade = self.test_suite.grade_answer(q, p_res.answer, p_res.citations)
            return ResultsCsvRow(
                request_id=p_res.request_id,
                scenario="burst",
                source=source,
                start_time=req_start_iso,
                first_token_ms=p_res.latency.first_token_ms,
                total_ms=p_res.latency.total_ms,
                status=p_res.status,
                retries=p_res.retries,
                tokens_in=p_res.tokens_in,
                tokens_out=p_res.tokens_out,
                retrieval_setting=p_res.retrieval_setting_name,
                answer_grade=grade,
            )

        with ThreadPoolExecutor(max_workers=min(10, burst_size)) as pool:
            futures = [pool.submit(worker, i) for i in range(burst_size)]
            for f in as_completed(futures):
                res.rows.append(f.result())

        res.duration_s = max(1.0, time.perf_counter() - start_time)
        res.requests_sent = burst_size
        res.offered_rate_per_min = round((burst_size / res.duration_s) * 60, 1)
        res.peak_queue_length = self.dispatcher.limiter.peak_queue_depth
        return res

    def run_sustained_overload(
        self,
        n_requests: int = 40,
        source: SourceType = SourceType.STAND_IN,
    ) -> ScenarioResult:
        """Scenario 3: Sustained overload exceeding RPM and triggering queue backpressure."""
        self.dispatcher.limiter.reset()
        res = ScenarioResult("sustained_overload", source)
        start_time = time.perf_counter()

        def worker(idx: int):
            # Stagger slightly by 50ms to create continuous arrival pressure
            time.sleep(idx * 0.05)
            q = self.questions[idx % len(self.questions)]
            req_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p_res = self.dispatcher.dispatch(q.question, scenario_name="sustained_overload")
            grade = self.test_suite.grade_answer(q, p_res.answer, p_res.citations)
            return ResultsCsvRow(
                request_id=p_res.request_id,
                scenario="sustained_overload",
                source=source,
                start_time=req_start_iso,
                first_token_ms=p_res.latency.first_token_ms,
                total_ms=p_res.latency.total_ms,
                status=p_res.status,
                retries=p_res.retries,
                tokens_in=p_res.tokens_in,
                tokens_out=p_res.tokens_out,
                retrieval_setting=p_res.retrieval_setting_name,
                answer_grade=grade,
            )

        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(worker, i) for i in range(n_requests)]
            for f in as_completed(futures):
                res.rows.append(f.result())

        res.duration_s = max(1.0, time.perf_counter() - start_time)
        res.requests_sent = n_requests
        res.offered_rate_per_min = round((n_requests / res.duration_s) * 60, 1)
        res.peak_queue_length = self.dispatcher.limiter.peak_queue_depth
        return res

    def run_per_minute_limit_halved(
        self,
        n_requests: int = 25,
        source: SourceType = SourceType.STAND_IN,
    ) -> ScenarioResult:
        """Scenario 4: Per-minute limit halved to 15 RPM, testing degraded behavior."""
        self.dispatcher.limiter.reset()
        self.dispatcher.limiter.halve_limits()
        res = ScenarioResult("per_minute_limit_halved", source)
        start_time = time.perf_counter()

        def worker(idx: int):
            time.sleep(idx * 0.1)
            q = self.questions[idx % len(self.questions)]
            req_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p_res = self.dispatcher.dispatch(q.question, scenario_name="per_minute_limit_halved")
            grade = self.test_suite.grade_answer(q, p_res.answer, p_res.citations)
            return ResultsCsvRow(
                request_id=p_res.request_id,
                scenario="per_minute_limit_halved",
                source=source,
                start_time=req_start_iso,
                first_token_ms=p_res.latency.first_token_ms,
                total_ms=p_res.latency.total_ms,
                status=p_res.status,
                retries=p_res.retries,
                tokens_in=p_res.tokens_in,
                tokens_out=p_res.tokens_out,
                retrieval_setting=p_res.retrieval_setting_name,
                answer_grade=grade,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(n_requests)]
            for f in as_completed(futures):
                res.rows.append(f.result())

        res.duration_s = max(1.0, time.perf_counter() - start_time)
        res.requests_sent = n_requests
        res.offered_rate_per_min = round((n_requests / res.duration_s) * 60, 1)
        res.peak_queue_length = self.dispatcher.limiter.peak_queue_depth
        return res

    def run_daily_budget_exhausted(
        self,
        n_requests: int = 15,
        source: SourceType = SourceType.STAND_IN,
    ) -> ScenarioResult:
        """Scenario 5: 3:00 PM scenario where daily token budget is exhausted."""
        self.dispatcher.limiter.reset()
        self.dispatcher.limiter.exhaust_daily_budget()
        res = ScenarioResult("daily_budget_exhausted", source)
        start_time = time.perf_counter()

        for i in range(n_requests):
            q = self.questions[i % len(self.questions)]
            req_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            p_res = self.dispatcher.dispatch(q.question, scenario_name="daily_budget_exhausted")
            grade = self.test_suite.grade_answer(q, p_res.answer, p_res.citations)

            row = ResultsCsvRow(
                request_id=p_res.request_id,
                scenario="daily_budget_exhausted",
                source=source,
                start_time=req_start_iso,
                first_token_ms=p_res.latency.first_token_ms,
                total_ms=p_res.latency.total_ms,
                status=p_res.status,
                retries=p_res.retries,
                tokens_in=p_res.tokens_in,
                tokens_out=p_res.tokens_out,
                retrieval_setting=p_res.retrieval_setting_name,
                answer_grade=grade,
            )
            res.rows.append(row)

        res.duration_s = max(0.1, time.perf_counter() - start_time)
        res.requests_sent = n_requests
        res.offered_rate_per_min = round((n_requests / res.duration_s) * 60, 1)
        res.peak_queue_length = self.dispatcher.limiter.peak_queue_depth
        return res

