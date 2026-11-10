"""Thread-safe rate limiters and telemetry state for the Stand-in model service."""

import time
import threading
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field


class RateLimiterConfig(BaseModel):
    """Configurable rate limits matching Groq free tier baseline."""
    requests_per_minute: int = 30
    tokens_per_minute: int = 6000
    requests_per_day: int = 1000
    tokens_per_day: int = 100000
    max_concurrency: int = 5


class RateLimiterState:
    """Sliding-window rate limiter enforcing RPM, TPM, RPD, and TPD."""

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        self.config = config or RateLimiterConfig()
        self._lock = threading.Lock()

        # Timestamps of requests in the current 60s window: list of (timestamp, token_count)
        self.minute_window: list[Tuple[float, int]] = []

        # Daily usage counters
        self.daily_requests_used: int = 0
        self.daily_tokens_used: int = 0
        self.daily_budget_exhausted: bool = False

        # In-flight tracking
        self.active_in_flight: int = 0
        self.peak_in_flight: int = 0

    def prune_minute_window(self, now: float) -> None:
        """Removes entries older than 60 seconds from the minute window."""
        cutoff = now - 60.0
        self.minute_window = [entry for entry in self.minute_window if entry[0] > cutoff]

    def check_and_consume(
        self, estimated_tokens: int, now: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """Checks if a request can proceed under current limits.

        Returns:
            (allowed: bool, reason: Optional[str], retry_after_seconds: Optional[float])
        """
        now = now or time.time()
        with self._lock:
            # 1. Check daily budget
            if self.daily_budget_exhausted or (
                self.daily_requests_used >= self.config.requests_per_day
            ) or (self.daily_tokens_used + estimated_tokens > self.config.tokens_per_day):
                return (
                    False,
                    "Daily quota exhausted. The daily budget has been reached. Resets at 00:00 UTC.",
                    86400.0,
                )

            # 2. Check minute window limits
            self.prune_minute_window(now)
            current_minute_reqs = len(self.minute_window)
            current_minute_tokens = sum(tokens for _, tokens in self.minute_window)

            if current_minute_reqs >= self.config.requests_per_minute:
                # Oldest request in window tells us when the first slot frees up
                oldest_time = self.minute_window[0][0]
                retry_after = max(1.0, round((oldest_time + 60.0) - now, 1))
                return (
                    False,
                    f"Rate limit reached for model: 30 requests/minute exceeded (current: {current_minute_reqs}).",
                    retry_after,
                )

            if current_minute_tokens + estimated_tokens > self.config.tokens_per_minute:
                oldest_time = self.minute_window[0][0]
                retry_after = max(1.0, round((oldest_time + 60.0) - now, 1))
                return (
                    False,
                    f"Rate limit reached for model: {self.config.tokens_per_minute} tokens/minute exceeded.",
                    retry_after,
                )

            # Request is accepted: update state
            self.minute_window.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.daily_tokens_used += estimated_tokens
            self.active_in_flight += 1
            if self.active_in_flight > self.peak_in_flight:
                self.peak_in_flight = self.active_in_flight

            return (True, None, None)

    def release_in_flight(self) -> None:
        """Decrements the active in-flight counter upon request completion."""
        with self._lock:
            if self.active_in_flight > 0:
                self.active_in_flight -= 1

    def halve_minute_limit(self) -> None:
        """Halves the per-minute limits for testing degraded scenarios."""
        with self._lock:
            self.config.requests_per_minute = max(1, self.config.requests_per_minute // 2)
            self.config.tokens_per_minute = max(100, self.config.tokens_per_minute // 2)

    def exhaust_daily_budget(self) -> None:
        """Immediately triggers daily budget exhaustion (e.g. 3pm scenario)."""
        with self._lock:
            self.daily_budget_exhausted = True
            self.daily_tokens_used = self.config.tokens_per_day
            self.daily_requests_used = self.config.requests_per_day

    def reset(self) -> None:
        """Resets all windows and counters."""
        with self._lock:
            self.minute_window.clear()
            self.daily_requests_used = 0
            self.daily_tokens_used = 0
            self.daily_budget_exhausted = False
            self.active_in_flight = 0
            self.peak_in_flight = 0


class StandinTelemetry:
    """Tracks cumulative service metrics across requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.rate_limited_requests: int = 0
        self.timed_out_requests: int = 0
        self.error_requests: int = 0
        self.retries_total: int = 0
        self.tokens_in_total: int = 0
        self.tokens_out_total: int = 0

    def record_outcome(
        self,
        status: str,
        retries: int,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.retries_total += retries
            self.tokens_in_total += tokens_in
            self.tokens_out_total += tokens_out

            if status == "ok":
                self.successful_requests += 1
            elif status == "rate_limited":
                self.rate_limited_requests += 1
            elif status == "timeout":
                self.timed_out_requests += 1
            else:
                self.error_requests += 1

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "succeeded": self.successful_requests,
                "rate_limited": self.rate_limited_requests,
                "timed_out": self.timed_out_requests,
                "errors": self.error_requests,
                "retries": self.retries_total,
                "tokens_in": self.tokens_in_total,
                "tokens_out": self.tokens_out_total,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.rate_limited_requests = 0
            self.timed_out_requests = 0
            self.error_requests = 0
            self.retries_total = 0
            self.tokens_in_total = 0
            self.tokens_out_total = 0
