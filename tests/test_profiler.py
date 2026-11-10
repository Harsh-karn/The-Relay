"""Tests for the Groq profiler and Table 1 generation."""

import pytest
from unittest.mock import patch, MagicMock
from the_relay.profiler.groq_profiler import GroqProfiler, GROQ_BASELINE_LIMITS
from the_relay.models import RequestStatus, SourceType


def test_baseline_limits_complete():
    """Verify all 7 required limits from Table 1 are defined with appropriate confidences."""
    names = [r.limit_name for r in GROQ_BASELINE_LIMITS]
    expected = [
        "Requests per minute",
        "Tokens per minute",
        "Requests per day",
        "Tokens per day",
        "Requests in flight at once",
        "What it returns when a limit is hit",
        "Time until it recovers",
    ]
    for exp in expected:
        assert exp in names
    for r in GROQ_BASELINE_LIMITS:
        assert r.confidence in ("high", "medium", "low")
        assert len(r.value_found) > 0
        assert len(r.how_found) > 0


def test_format_table1_markdown():
    """Verify Table 1 renders valid markdown matching assignment heading requirements."""
    md = GroqProfiler.format_table1_markdown(GROQ_BASELINE_LIMITS)
    assert "### Table 1: Limits you found in the real service" in md
    assert "| Requests per minute |" in md
    assert "| Tokens per minute |" in md
    assert "| Requests in flight at once |" in md


def test_profiler_without_key_raises_informative_error():
    """Without an API key, sending a live request should raise an explicit ValueError."""
    profiler = GroqProfiler(api_key="")
    assert not profiler.is_live_available
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        profiler.send_single_streaming_request("test")


def test_profiler_mocked_streaming_request():
    """Simulate a live SSE response from Groq with rate limit headers."""
    profiler = GroqProfiler(api_key="gsk_mock_test_key_for_profiler_unit_tests")
    assert profiler.is_live_available

    mock_sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Attention "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "is "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "all "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "you need."}}]}\n',
        b'data: [DONE]\n',
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-limit-tokens": "6000",
        "x-ratelimit-remaining-requests": "29",
        "x-ratelimit-remaining-tokens": "5950",
    }
    mock_response.iter_lines.return_value = [line.decode("utf-8") for line in mock_sse_lines]
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client):
        resp = profiler.send_single_streaming_request("Explain attention")
        assert resp.status == RequestStatus.OK
        assert resp.source == SourceType.REAL
        assert resp.text == "Attention is all you need."
        assert resp.tokens_out == 4
        assert resp.first_token_ms > 0
        assert resp.rate_limit_headers.get("x-ratelimit-limit-requests") == "30"


def test_profiler_mocked_429_rate_limit():
    """Verify 429 responses are captured as RATE_LIMITED status with retry-after header."""
    profiler = GroqProfiler(api_key="gsk_mock_test_key")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {
        "retry-after": "12",
        "x-ratelimit-reset-requests": "12s",
    }
    mock_response.read.return_value = b'{"error": {"message": "Rate limit exceeded", "type": "requests", "code": "rate_limit_exceeded"}}'
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client):
        resp = profiler.send_single_streaming_request("Test rate limit")
        assert resp.status == RequestStatus.RATE_LIMITED
        assert resp.rate_limit_headers.get("retry-after") == "12"
        assert "Rate limit exceeded" in resp.error_message
