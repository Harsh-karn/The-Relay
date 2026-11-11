"""Retry Policy with Exponential Backoff, Full Jitter, and Systemic Retry Budget."""

import time
import random
import threading
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel


class RetryConfig(BaseModel):
    max_retries: int = 2
    base_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    retry_budget_ratio: float = 0.20  # Max 20% of requests can be retried system-wide
    min_budget_pool: int = 5


class RetryManager:
    """Controls request retries with exponential backoff, jitter, and global retry budgeting."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._lock = threading.Lock()

        # System-wide retry budget tracking
        self.total_attempts: int = 0
        self.retries_consumed: int = 0

    def can_retry(
        self,
        attempt_number: int,
        status_code_or_reason: str,
        elapsed_request_s: float,
        client_timeout_s: float,
    ) -> Tuple[bool, str]:
        """Determines if a retry should be permitted.

        Returns:
            (allow: bool, reason: str)
        """
        # 1. Check max per-request limit
        if attempt_number >= self.config.max_retries:
            return (False, f"Max retries ({self.config.max_retries}) reached")

        # 2. Check if remaining client timeout is too short for a backoff
        if elapsed_request_s + self.config.base_backoff_s >= client_timeout_s:
            return (False, "Insufficient time before client timeout")

        # 3. Check for non-retryable reasons
        non_retryable = [
            "DAILY_BUDGET_EXHAUSTED",
            "400_BAD_REQUEST",
            "401_UNAUTHORIZED",
            "CLIENT_TIMEOUT",
        ]
        if any(nr in status_code_or_reason.upper() for nr in non_retryable):
            return (False, f"Non-retryable status: {status_code_or_reason}")

        # 4. Check global retry budget
        with self._lock:
            allowed_retries = max(
                self.config.min_budget_pool,
                int(self.total_attempts * self.config.retry_budget_ratio),
            )
            if self.retries_consumed >= allowed_retries:
                return (False, "Systemic retry budget exhausted")

            # Admit retry
            self.retries_consumed += 1
            return (True, "Retry admitted")

    def record_attempt(self) -> None:
        """Records a new initial request attempt."""
        with self._lock:
            self.total_attempts += 1

    def calculate_backoff(
        self,
        attempt_number: int,
        upstream_retry_after: Optional[float] = None,
    ) -> float:
        """Calculates exponential backoff with AWS Full Jitter algorithm."""
        base = self.config.base_backoff_s
        cap = self.config.max_backoff_s
        calculated_backoff = min(cap, base * (2 ** attempt_number))

        # Full jitter: random between 0 and calculated_backoff
        jittered = random.uniform(0.1, calculated_backoff)

        # If upstream gave an explicit retry-after header, honor it
        if upstream_retry_after and upstream_retry_after > 0:
            return max(jittered, upstream_retry_after)

        return jittered

    def reset(self) -> None:
        with self._lock:
            self.total_attempts = 0
            self.retries_consumed = 0

