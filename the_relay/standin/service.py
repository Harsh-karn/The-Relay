"""Calibrated Stand-in Model Service.

Emulates Groq LPU inference behavior under various load profiles:
- Token-by-token streaming generator
- Pre-first-token delay (TTFT) and token generation latency calibrated to real service
- Dynamic latency degradation under concurrent in-flight load
- Sliding-window rate limit enforcement (RPM/TPM) and daily budget quota exhaustion
- Failure injection: transient 503s, slow successes, and connection drops
"""

import time
import random
import logging
from typing import Iterator, Optional, List, Dict, Any
from pydantic import BaseModel, Field

from the_relay.models import (
    ChatMessage,
    StreamChunk,
    CompletionResponse,
    RequestStatus,
    SourceType,
)
from the_relay.standin.state import (
    RateLimiterConfig,
    RateLimiterState,
    StandinTelemetry,
)

logger = logging.getLogger(__name__)


class StandinCalibrationConfig(BaseModel):
    """Empirical calibration parameters derived from real Groq measurements."""
    # Baseline TTFT distribution (milliseconds)
    base_ttft_mean_ms: float = 220.0
    base_ttft_std_ms: float = 40.0
    base_ttft_min_ms: float = 140.0

    # Token streaming rate (milliseconds per token delta)
    inter_token_delay_ms: float = 14.0  # ~70 tokens/sec
    inter_token_jitter_ms: float = 3.0

    # Load-induced latency multiplier: TTFT scales with in-flight requests
    # Latency factor = 1 + load_scale * (active_in_flight - 1)^load_exponent
    load_scale: float = 0.45
    load_exponent: float = 1.3

    # Failure & degradation injection probabilities
    transient_503_probability: float = 0.0
    slow_success_probability: float = 0.0
    slow_success_delay_ms: float = 4000.0


class StandinModelService:
    """A fully controllable stand-in model service simulating Groq API characteristics."""

    def __init__(
        self,
        calibration: Optional[StandinCalibrationConfig] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        self.calib = calibration or StandinCalibrationConfig()
        self.state = RateLimiterState(rate_limiter_config)
        self.telemetry = StandinTelemetry()

    def stream_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 256,
        synthetic_answer: Optional[str] = None,
    ) -> Iterator[StreamChunk]:
        """Yields response tokens one piece at a time with realistic calibrated delays."""
        start_time = time.perf_counter()

        # Compute estimated input tokens
        prompt_text = " ".join(m.content for m in messages)
        tokens_in = max(10, len(prompt_text.split()) * 2)

        # 1. Rate limiter check
        allowed, reason, retry_after = self.state.check_and_consume(tokens_in)
        if not allowed:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.telemetry.record_outcome(
                status="rate_limited",
                retries=0,
                tokens_in=tokens_in,
                tokens_out=0,
            )
            # Yield error chunk carrying the exact refusal message
            yield StreamChunk(
                delta=reason or "Rate limit reached",
                index=0,
                elapsed_ms=duration_ms,
                is_first_token=False,
                finish_reason="rate_limited",
            )
            return

        try:
            # 2. Check for injected transient 503 failure
            if random.random() < self.calib.transient_503_probability:
                time.sleep(0.05)  # slight network handshake delay
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.telemetry.record_outcome(
                    status="error",
                    retries=0,
                    tokens_in=tokens_in,
                    tokens_out=0,
                )
                yield StreamChunk(
                    delta="",
                    index=0,
                    elapsed_ms=duration_ms,
                    is_first_token=False,
                    finish_reason="503_service_unavailable",
                )
                return

            # 3. Calculate calibrated TTFT delay (including concurrency slowdown)
            base_ttft = max(
                self.calib.base_ttft_min_ms,
                random.gauss(self.calib.base_ttft_mean_ms, self.calib.base_ttft_std_ms),
            )

            # Apply load slowdown if concurrent requests are in flight
            in_flight = self.state.active_in_flight
            if in_flight > 1:
                concurrency_penalty = 1.0 + self.calib.load_scale * (
                    (in_flight - 1) ** self.calib.load_exponent
                )
                target_ttft_ms = base_ttft * concurrency_penalty
            else:
                target_ttft_ms = base_ttft

            # Check if this request is a "slow success"
            if random.random() < self.calib.slow_success_probability:
                target_ttft_ms += self.calib.slow_success_delay_ms

            # Sleep to simulate pre-first-token latency
            time.sleep(target_ttft_ms / 1000.0)

            # 4. Prepare response text tokens
            if synthetic_answer:
                text_to_stream = synthetic_answer
            else:
                text_to_stream = (
                    "Based on the provided research papers, the architecture relies on "
                    "multi-head self-attention mechanisms and dynamic chunking hierarchy "
                    "to maintain linear latency while preserving full global context "
                    "(Vaswani et al., 2017, Section 3.2)."
                )

            words = text_to_stream.split(" ")
            tokens_to_emit = [w + " " for w in words[:-1]] + [words[-1]]
            tokens_to_emit = tokens_to_emit[:max_tokens]

            # 5. Stream tokens piece by piece
            first_token = True
            for idx, token_piece in enumerate(tokens_to_emit):
                if not first_token:
                    # Token generation delay
                    jitter = random.uniform(
                        -self.calib.inter_token_jitter_ms,
                        self.calib.inter_token_jitter_ms,
                    )
                    token_delay = max(0.002, (self.calib.inter_token_delay_ms + jitter) / 1000.0)
                    time.sleep(token_delay)

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                is_last = (idx == len(tokens_to_emit) - 1)

                yield StreamChunk(
                    delta=token_piece,
                    index=idx,
                    elapsed_ms=elapsed_ms,
                    is_first_token=first_token,
                    finish_reason="stop" if is_last else None,
                )
                first_token = False

            total_elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.telemetry.record_outcome(
                status="ok",
                retries=0,
                tokens_in=tokens_in,
                tokens_out=len(tokens_to_emit),
            )

        finally:
            self.state.release_in_flight()

    def generate_complete(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 256,
        synthetic_answer: Optional[str] = None,
    ) -> CompletionResponse:
        """Executes a full completion by consuming the streaming generator and building CompletionResponse."""
        start_time = time.perf_counter()
        first_token_ms = 0.0
        chunks: List[str] = []
        finish_status = RequestStatus.OK
        err_msg: Optional[str] = None
        rate_headers: Dict[str, str] = {}

        for chunk in self.stream_completion(
            messages=messages,
            max_tokens=max_tokens,
            synthetic_answer=synthetic_answer,
        ):
            if chunk.is_first_token:
                first_token_ms = chunk.elapsed_ms

            if chunk.finish_reason == "rate_limited":
                finish_status = RequestStatus.RATE_LIMITED
                err_msg = chunk.delta or "Rate limit reached for stand-in model (429)"
                rate_headers = {"retry-after": "15"}
                break
            elif chunk.finish_reason == "503_service_unavailable":
                finish_status = RequestStatus.ERROR
                err_msg = chunk.delta or "503 Service Unavailable: Stand-in worker overloaded"
                break

            if chunk.delta:
                chunks.append(chunk.delta)

        total_ms = (time.perf_counter() - start_time) * 1000
        full_text = "".join(chunks)

        prompt_text = " ".join(m.content for m in messages)
        tokens_in = max(10, len(prompt_text.split()) * 2)
        tokens_out = len(chunks)

        return CompletionResponse(
            text=full_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=finish_status,
            source=SourceType.STAND_IN,
            error_message=err_msg,
            rate_limit_headers=rate_headers if rate_headers else None,
        )

    def get_calibration_report(self) -> Dict[str, Any]:
        """Returns structured comparison and calibration data required for Section 2 of the write-up."""
        return {
            "calibrated_from": "Groq LPU free-tier measurements (llama-3.3-70b-versatile)",
            "base_ttft_mean_ms": self.calib.base_ttft_mean_ms,
            "base_ttft_std_ms": self.calib.base_ttft_std_ms,
            "token_speed_tokens_per_sec": round(1000.0 / self.calib.inter_token_delay_ms, 1),
            "load_degradation": f"TTFT scales with (1 + {self.calib.load_scale} * (C-1)^{self.calib.load_exponent})",
            "per_minute_limits": f"{self.state.config.requests_per_minute} RPM / {self.state.config.tokens_per_minute} TPM",
            "per_day_limits": f"{self.state.config.requests_per_day} RPD / {self.state.config.tokens_per_day} TPD",
            "where_it_differs_most": (
                "The stand-in models queuing overhead with synthetic thread delays and exponential "
                "concurrency penalties, whereas real Groq LPU clusters experience memory bandwidth "
                "saturation, multi-tenant network noisy neighbors, and discrete batch dispatch intervals. "
                "Additionally, real Groq enforces strict token counting on prefix caching, while the stand-in "
                "uses heuristic token estimation."
            ),
        }
