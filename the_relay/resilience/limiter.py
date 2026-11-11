"""Adaptive Rate Limiter, Concurrency Gate, and Quota Controller.

Enforces:
- Sliding-window Token & Request Rate Limiting (RPM, TPM)
- In-flight Concurrency Gate (preventing server saturation)
- Daily Quota Shedding (3:00 PM budget exhaustion policy)
- Queue Depth Shedding (dropping low-priority requests when queue is saturated)
"""

import time
import threading
from typing import Tuple, Optional, List
from pydantic import BaseModel


class RateLimiterPolicy(BaseModel):
    requests_per_minute: int = 30
    tokens_per_minute: int = 6000
    daily_token_limit: int = 100000
    daily_request_limit: int = 1000
    max_in_flight: int = 3           # Empirical Groq concurrency cap
    max_queue_depth: int = 25        # Beyond this, drop requests with fast 429
    client_timeout_s: float = 12.0   # Users give up waiting after this window


class AdaptiveRateLimiter:
    """Thread-safe rate limiter managing rate limits, concurrency gates, and daily quota."""

    def __init__(self, policy: Optional[RateLimiterPolicy] = None):
        self.policy = policy or RateLimiterPolicy()
        self._lock = threading.Lock()

        # Minute window: list of (timestamp, token_count)
        self.minute_window: List[Tuple[float, int]] = []

        # Daily usage
        self.daily_tokens_used: int = 0
        self.daily_requests_used: int = 0
        self.daily_exhausted: bool = False

        # In-flight concurrency gate
        self.in_flight: int = 0
        self.peak_in_flight: int = 0

        # Queue tracking
        self.current_queue_depth: int = 0
        self.peak_queue_depth: int = 0

    def prune_window(self, now: float) -> None:
        cutoff = now - 60.0
        self.minute_window = [e for e in self.minute_window if e[0] > cutoff]

    def acquire_slot(
        self,
        estimated_tokens: int = 200,
        now: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """Checks if a request can be dispatched immediately.

        Returns:
            (can_dispatch: bool, rejection_reason: Optional[str], wait_time: Optional[float])
        """
        now = now or time.time()
        with self._lock:
            # 1. Check daily budget exhaustion (immediate shedding)
            if self.daily_exhausted or (
                self.daily_requests_used >= self.policy.daily_request_limit
            ) or (self.daily_tokens_used + estimated_tokens > self.policy.daily_token_limit):
                self.daily_exhausted = True
                return (
                    False,
                    "DAILY_BUDGET_EXHAUSTED",
                    86400.0,
                )

            # 2. Check in-flight concurrency
            if self.in_flight >= self.policy.max_in_flight:
                # Concurrency saturated, must queue or wait
                return (
                    False,
                    "CONCURRENCY_SATURATED",
                    0.25,
                )

            # 3. Check sliding minute window (RPM / TPM)
            self.prune_window(now)
            current_reqs = len(self.minute_window)
            current_tokens = sum(t for _, t in self.minute_window)

            if current_reqs >= self.policy.requests_per_minute:
                oldest_ts = self.minute_window[0][0]
                retry_after = max(0.5, round((oldest_ts + 60.0) - now, 2))
                return (
                    False,
                    "RPM_EXCEEDED",
                    retry_after,
                )

            if current_tokens + estimated_tokens > self.policy.tokens_per_minute:
                oldest_ts = self.minute_window[0][0]
                retry_after = max(0.5, round((oldest_ts + 60.0) - now, 2))
                return (
                    False,
                    "TPM_EXCEEDED",
                    retry_after,
                )

            # Request admitted: consume resource
            self.minute_window.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.daily_tokens_used += estimated_tokens
            self.in_flight += 1
            if self.in_flight > self.peak_in_flight:
                self.peak_in_flight = self.in_flight

            return (True, None, None)

    def release_slot(self) -> None:
        """Frees an in-flight execution slot."""
        with self._lock:
            if self.in_flight > 0:
                self.in_flight -= 1

    def enter_queue(self) -> Tuple[bool, int]:
        """Registers a request entering the queue; drops if queue is full."""
        with self._lock:
            if self.current_queue_depth >= self.policy.max_queue_depth:
                return (False, self.current_queue_depth)
            self.current_queue_depth += 1
            if self.current_queue_depth > self.peak_queue_depth:
                self.peak_queue_depth = self.current_queue_depth
            return (True, self.current_queue_depth)

    def leave_queue(self) -> None:
        with self._lock:
            if self.current_queue_depth > 0:
                self.current_queue_depth -= 1

    def halve_limits(self) -> None:
        """Halves per-minute limits for testing degraded scenarios."""
        with self._lock:
            self.policy.requests_per_minute = max(1, self.policy.requests_per_minute // 2)
            self.policy.tokens_per_minute = max(100, self.policy.tokens_per_minute // 2)

    def exhaust_daily_budget(self) -> None:
        """Immediately exhausts daily budget (3:00 PM scenario)."""
        with self._lock:
            self.daily_exhausted = True
            self.daily_tokens_used = self.policy.daily_token_limit
            self.daily_requests_used = self.policy.daily_request_limit

    def reset(self) -> None:
        with self._lock:
            self.minute_window.clear()
            self.daily_tokens_used = 0
            self.daily_requests_used = 0
            self.daily_exhausted = False
            self.in_flight = 0
            self.peak_in_flight = 0
            self.current_queue_depth = 0
            self.peak_queue_depth = 0
