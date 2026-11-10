"""Unit tests for the Stand-in model service and unified client."""

import pytest
import time
from the_relay.models import (
    ChatMessage,
    RequestStatus,
    SourceType,
)
from the_relay.standin.state import RateLimiterConfig
from the_relay.standin.service import (
    StandinModelService,
    StandinCalibrationConfig,
)
from the_relay.client import RelayModelClient


@pytest.fixture
def fast_calibration():
    """Fast calibration config for rapid unit testing without long delays."""
    return StandinCalibrationConfig(
        base_ttft_mean_ms=10.0,
        base_ttft_std_ms=2.0,
        base_ttft_min_ms=5.0,
        inter_token_delay_ms=2.0,
        inter_token_jitter_ms=0.5,
        load_scale=0.5,
        load_exponent=1.2,
        transient_503_probability=0.0,
        slow_success_probability=0.0,
    )


def test_standin_streaming_token_by_token(fast_calibration):
    """Verify that stand-in yields tokens piece by piece and tracks TTFT."""
    service = StandinModelService(calibration=fast_calibration)
    messages = [ChatMessage(role="user", content="Explain attention.")]

    chunks = list(service.stream_completion(messages, synthetic_answer="Self attention is all you need."))
    assert len(chunks) > 0
    assert any(c.is_first_token for c in chunks)
    first_chunk = next(c for c in chunks if c.is_first_token)
    assert first_chunk.elapsed_ms > 0
    full_text = "".join(c.delta for c in chunks)
    assert full_text == "Self attention is all you need."


def test_standin_generate_complete(fast_calibration):
    """Verify full completion response schema and latency fields."""
    service = StandinModelService(calibration=fast_calibration)
    messages = [ChatMessage(role="user", content="Test query")]

    resp = service.generate_complete(messages, synthetic_answer="Verified response.")
    assert resp.status == RequestStatus.OK
    assert resp.source == SourceType.STAND_IN
    assert resp.text == "Verified response."
    assert resp.first_token_ms > 0
    assert resp.total_ms >= resp.first_token_ms
    assert resp.tokens_out > 0


def test_standin_load_slowdown(fast_calibration):
    """Verify that active in-flight requests increase target TTFT."""
    service = StandinModelService(calibration=fast_calibration)

    # Artificially set in-flight requests
    service.state.active_in_flight = 1
    messages = [ChatMessage(role="user", content="Low load")]
    resp_low = service.generate_complete(messages, synthetic_answer="Low load answer")

    service.state.active_in_flight = 6
    messages = [ChatMessage(role="user", content="High load")]
    resp_high = service.generate_complete(messages, synthetic_answer="High load answer")

    # High load TTFT should be significantly higher due to concurrency penalty
    assert resp_high.first_token_ms > resp_low.first_token_ms


def test_standin_per_minute_rate_limit(fast_calibration):
    """Verify that exceeding RPM (e.g., 5 in small config) triggers rate_limited status."""
    small_limits = RateLimiterConfig(requests_per_minute=3, tokens_per_minute=10000)
    service = StandinModelService(
        calibration=fast_calibration, rate_limiter_config=small_limits
    )
    messages = [ChatMessage(role="user", content="Short prompt")]

    # First 3 should succeed
    for _ in range(3):
        resp = service.generate_complete(messages, synthetic_answer="OK")
        assert resp.status == RequestStatus.OK

    # 4th request must be rate limited
    resp_blocked = service.generate_complete(messages, synthetic_answer="Blocked")
    assert resp_blocked.status == RequestStatus.RATE_LIMITED
    assert resp_blocked.rate_limit_headers is not None
    assert "retry-after" in resp_blocked.rate_limit_headers


def test_standin_halve_minute_limit(fast_calibration):
    """Verify dynamic halving of per-minute limit."""
    limits = RateLimiterConfig(requests_per_minute=4)
    service = StandinModelService(
        calibration=fast_calibration, rate_limiter_config=limits
    )
    service.state.halve_minute_limit()
    assert service.state.config.requests_per_minute == 2

    messages = [ChatMessage(role="user", content="Prompt")]
    # First 2 succeed
    assert service.generate_complete(messages, synthetic_answer="1").status == RequestStatus.OK
    assert service.generate_complete(messages, synthetic_answer="2").status == RequestStatus.OK
    # 3rd is blocked
    assert service.generate_complete(messages, synthetic_answer="3").status == RequestStatus.RATE_LIMITED


def test_standin_daily_budget_exhaustion(fast_calibration):
    """Verify immediate refusal when daily budget is exhausted (3pm scenario)."""
    service = StandinModelService(calibration=fast_calibration)
    service.state.exhaust_daily_budget()

    messages = [ChatMessage(role="user", content="Afternoon question")]
    resp = service.generate_complete(messages)
    assert resp.status == RequestStatus.RATE_LIMITED
    assert "Daily quota exhausted" in (resp.error_message or "")


def test_standin_injected_503_failure():
    """Verify injected transient failure behavior."""
    fail_calibration = StandinCalibrationConfig(
        base_ttft_mean_ms=5.0,
        transient_503_probability=1.0,  # 100% failure rate
    )
    service = StandinModelService(calibration=fail_calibration)
    messages = [ChatMessage(role="user", content="Fail me")]

    resp = service.generate_complete(messages)
    assert resp.status == RequestStatus.ERROR
    assert "503" in (resp.error_message or "")


def test_calibration_report_contains_differences():
    """Verify stand-in calibration report explicitly specifies where it differs from real Groq."""
    service = StandinModelService()
    report = service.get_calibration_report()
    assert "where_it_differs_most" in report
    assert len(report["where_it_differs_most"]) > 50
    assert "LPU" in report["where_it_differs_most"]


def test_relay_model_client_dispatch(fast_calibration):
    """Verify unified client routes properly to stand-in service."""
    service = StandinModelService(calibration=fast_calibration)
    client = RelayModelClient(source=SourceType.STAND_IN, standin_service=service)

    messages = [ChatMessage(role="user", content="Client test")]
    resp = client.complete(messages, synthetic_answer="Routed via stand-in.")
    assert resp.status == RequestStatus.OK
    assert resp.source == SourceType.STAND_IN
    assert resp.text == "Routed via stand-in."
