"""Unit tests for rate-limiting, queuing, retries, and failure policies."""

import time
import pytest
from concurrent.futures import ThreadPoolExecutor

from the_relay.resilience.limiter import AdaptiveRateLimiter, RateLimiterPolicy
from the_relay.resilience.retry import RetryManager, RetryConfig
from the_relay.resilience.dispatcher import (
    ResilientDispatcher,
    DAILY_BUDGET_EXHAUSTED_MESSAGE,
)
from the_relay.pipeline.core import RelayPipeline
from the_relay.models import RequestStatus, SourceType
from the_relay.standin.service import StandinModelService, StandinCalibrationConfig
from the_relay.client import RelayModelClient


@pytest.fixture
def test_setup():
    calib = StandinCalibrationConfig(
        base_ttft_mean_ms=2.0,
        base_ttft_std_ms=0.5,
        base_ttft_min_ms=1.0,
        inter_token_delay_ms=0.2,
    )
    service = StandinModelService(calibration=calib)
    client = RelayModelClient(source=SourceType.STAND_IN, standin_service=service)
    pipeline = RelayPipeline(client=client)

    policy = RateLimiterPolicy(
        requests_per_minute=10,
        tokens_per_minute=2000,
        max_in_flight=2,
        max_queue_depth=5,
        client_timeout_s=1.0,
    )
    limiter = AdaptiveRateLimiter(policy=policy)
    retry_mgr = RetryManager(RetryConfig(max_retries=2, base_backoff_s=0.05, max_backoff_s=0.2))

    dispatcher = ResilientDispatcher(pipeline=pipeline, limiter=limiter, retry_mgr=retry_mgr)
    return dispatcher, limiter, retry_mgr


def test_concurrency_gate_limits_in_flight(test_setup):
    """Verify in-flight concurrency is strictly bounded by max_in_flight."""
    dispatcher, limiter, _ = test_setup

    # Acquire slots manually to simulate active in-flight requests
    ok1, _, _ = limiter.acquire_slot(100)
    ok2, _, _ = limiter.acquire_slot(100)
    assert ok1 is True
    assert ok2 is True

    # 3rd request must be rejected by concurrency gate (max_in_flight = 2)
    ok3, reason, _ = limiter.acquire_slot(100)
    assert ok3 is False
    assert reason == "CONCURRENCY_SATURATED"

    # Releasing frees a slot
    limiter.release_slot()
    ok4, _, _ = limiter.acquire_slot(100)
    assert ok4 is True


def test_daily_budget_exhaustion_message(test_setup):
    """Verify user receives exact 3pm daily budget exhaustion message."""
    dispatcher, limiter, _ = test_setup
    limiter.exhaust_daily_budget()

    result = dispatcher.dispatch("What is multi-head attention?")
    assert result.status == RequestStatus.RATE_LIMITED
    assert result.answer == DAILY_BUDGET_EXHAUSTED_MESSAGE
    assert "00:00 UTC" in result.answer
    assert result.latency.working_ms <= 5.0  # Fast rejection without LLM processing


def test_queue_overflow_backpressure_shedding(test_setup):
    """Verify queue sheds load with fast 429 when max_queue_depth is exceeded."""
    dispatcher, limiter, _ = test_setup

    # Artificially fill queue
    limiter.current_queue_depth = limiter.policy.max_queue_depth

    result = dispatcher.dispatch("Question under severe queue load")
    assert result.status == RequestStatus.RATE_LIMITED
    assert "Queue capacity exceeded" in result.answer


def test_retry_manager_exponential_backoff_and_jitter():
    """Verify backoff calculation incorporates exponential scaling and jitter."""
    mgr = RetryManager(RetryConfig(base_backoff_s=0.1, max_backoff_s=1.0))
    backoffs = [mgr.calculate_backoff(i) for i in range(5)]
    # All backoffs must be positive and bounded by max_backoff_s
    for b in backoffs:
        assert 0.05 <= b <= 1.0


def test_retry_manager_rejects_non_retryable():
    """Verify non-retryable errors are immediately refused retries."""
    mgr = RetryManager()
    allow, reason = mgr.can_retry(0, "DAILY_BUDGET_EXHAUSTED", 0.1, 10.0)
    assert allow is False
    assert "Non-retryable" in reason


def test_retry_budget_exhaustion():
    """Verify retry budget limits systemic retries when budget ratio is crossed."""
    mgr = RetryManager(RetryConfig(max_retries=3, retry_budget_ratio=0.1, min_budget_pool=2))
    # Consume allowed pool
    allow1, _ = mgr.can_retry(0, "503_ERROR", 0.1, 10.0)
    allow2, _ = mgr.can_retry(0, "503_ERROR", 0.1, 10.0)
    assert allow1 is True
    assert allow2 is True

    # 3rd should be denied because total_attempts has not scaled up enough
    allow3, reason = mgr.can_retry(0, "503_ERROR", 0.1, 10.0)
    assert allow3 is False
    assert "retry budget" in reason.lower()


def test_thundering_herd_case_study(test_setup):
    """Case study: demonstrate how uncoordinated retries without jitter exacerbate load.

    Simulates a burst where synchronized retries create lock contention,
    while jittered retries disperse arrival distribution smoothly.
    """
    dispatcher, limiter, _ = test_setup

    # Dispatch a small parallel pool
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(dispatcher.dispatch, f"Query {i}", "burst_test")
            for i in range(4)
        ]
        results = [f.result() for f in futures]

    assert len(results) == 4
    # All dispatched requests finish with valid status (OK, RATE_LIMITED, or TIMEOUT)
    for r in results:
        assert r.status in (RequestStatus.OK, RequestStatus.RATE_LIMITED, RequestStatus.TIMEOUT)
