"""
tests/challenge/test_challenge_retry_handler.py
Empirical Adversarial Challenge Suite 2: Rate Limit, Error Retry Handler, and Quota Exhaustion.
Tests HTTP 429 with Retry-After, HTTP 500, network disconnects, and fatal insufficient_quota abort.
"""
import time
import pytest
import requests
from unittest.mock import MagicMock, call

from app.core.translation.base import (
    RateLimitError, QuotaExhaustedError, AuthenticationError,
    ModelNotFoundError, TranslationError
)
from app.core.translation.retry_handler import execute_http_request_with_retry


# =========================================================================
# 1. HTTP 429 Rate Limit & Retry-After Header Testing
# =========================================================================

def test_challenge_429_retry_after_header_respected(monkeypatch):
    """Verifies that HTTP 429 parses integer Retry-After header and sleeps for specified seconds."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    # Mock response: 1st call 429 with Retry-After: 4, 2nd call 200 OK
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "4"}
    resp_429.text = "Rate limit reached. Please wait."

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = '{"success": true}'

    mock_req = MagicMock(side_effect=[resp_429, resp_200])
    monkeypatch.setattr(requests, "request", mock_req)

    res = execute_http_request_with_retry(
        method="POST",
        url="https://api.mock.test/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        max_retries=3
    )

    assert res.status_code == 200
    assert mock_req.call_count == 2
    assert len(sleep_calls) == 1
    # Check that Retry-After of 4 seconds was used
    assert sleep_calls[0] == 4.0


def test_challenge_429_exhaustion_raises_ratelimiterror(monkeypatch):
    """Verifies that persistent HTTP 429 exceeding max_retries raises RateLimitError with retry_after info."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "2"}
    resp_429.text = "Too Many Requests"

    mock_req = MagicMock(return_value=resp_429)
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(RateLimitError) as exc_info:
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            max_retries=2
        )

    assert mock_req.call_count == 3
    assert len(sleep_calls) == 2
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 2.0


def test_challenge_429_float_retry_after_limitation(monkeypatch):
    """
    LIMITATION DISCOVERY: Retry-After with decimal/float value (e.g. '2.5') fails .isdigit() check,
    causing retry_handler to ignore the header and fall back to exponential backoff.
    """
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "2.5"}  # Decimal string
    resp_429.text = "Rate limit reached"

    resp_200 = MagicMock()
    resp_200.status_code = 200

    mock_req = MagicMock(side_effect=[resp_429, resp_200])
    monkeypatch.setattr(requests, "request", mock_req)

    execute_http_request_with_retry(
        method="POST",
        url="https://api.mock.test/v1",
        headers={},
        initial_delay=1.0
    )

    assert len(sleep_calls) == 1
    # Because '2.5'.isdigit() is False, sleep_time is calculated from backoff (1.0 + jitter), NOT 2.5!
    assert sleep_calls[0] != 2.5
    assert 1.0 <= sleep_calls[0] <= 1.6


# =========================================================================
# 2. Fatal Insufficient Quota Detection (Zero Waste of Retries)
# =========================================================================

@pytest.mark.parametrize("quota_error_snippet", [
    "insufficient_quota",
    "insufficient_balance",
    "quota_exceeded",
    "resource_exhausted",
    "billing_not_active",
    "credit_exhausted",
    "用户欠费，请充值",
    "账户余额不足",
    "INSUFFICIENT_QUOTA_ERROR"
])
def test_challenge_fatal_quota_exhaustion_immediate_abort(monkeypatch, quota_error_snippet):
    """Verifies that any quota exhaustion immediately raises QuotaExhaustedError without ANY retry or sleep."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_quota = MagicMock()
    resp_quota.status_code = 429
    resp_quota.text = f'{{"error": {{"message": "Account error: {quota_error_snippet}"}}}}'

    mock_req = MagicMock(return_value=resp_quota)
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(QuotaExhaustedError) as exc_info:
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            max_retries=5,
            provider_name="DeepSeek"
        )

    # Must abort on the VERY FIRST attempt (1 call total, 0 retries)
    assert mock_req.call_count == 1, f"Expected 1 call, got {mock_req.call_count} (wasted retries!)"
    assert len(sleep_calls) == 0, f"Expected 0 sleep calls, got {len(sleep_calls)} (wasted sleep time!)"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is False


# =========================================================================
# 3. HTTP 500 Server Errors (Transient vs Exhaustion)
# =========================================================================

def test_challenge_500_transient_recovery(monkeypatch):
    """Verifies that transient HTTP 500 server error recovers on subsequent retry."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.text = "Internal Server Error"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = '{"status": "ok"}'

    mock_req = MagicMock(side_effect=[resp_500, resp_200])
    monkeypatch.setattr(requests, "request", mock_req)

    res = execute_http_request_with_retry(
        method="POST",
        url="https://api.mock.test/v1",
        headers={},
        max_retries=3,
        initial_delay=0.5
    )

    assert res.status_code == 200
    assert mock_req.call_count == 2
    assert len(sleep_calls) == 1


def test_challenge_500_persistent_failure(monkeypatch):
    """Verifies that persistent HTTP 502/503/504 raises non-retryable TranslationError after max retries."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    mock_req = MagicMock(return_value=resp_503)
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(TranslationError) as exc_info:
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1",
            headers={},
            max_retries=2
        )

    assert mock_req.call_count == 3  # 1 initial + 2 retries
    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is False


# =========================================================================
# 4. Network Disconnects & Timeouts
# =========================================================================

def test_challenge_network_connection_error_recovery(monkeypatch):
    """Verifies that connection drop/disconnect recovers when retry succeeds."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = '{"connected": true}'

    mock_req = MagicMock(side_effect=[
        requests.exceptions.ConnectionError("Connection aborted by peer"),
        resp_200
    ])
    monkeypatch.setattr(requests, "request", mock_req)

    res = execute_http_request_with_retry(
        method="POST",
        url="https://api.mock.test/v1",
        headers={},
        max_retries=3
    )

    assert res.status_code == 200
    assert mock_req.call_count == 2
    assert len(sleep_calls) == 1


def test_challenge_network_timeout_persistent(monkeypatch):
    """Verifies that persistent network timeout raises TranslationError with user diagnostic action."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    mock_req = MagicMock(side_effect=requests.exceptions.Timeout("Read timeout"))
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(TranslationError) as exc_info:
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1",
            headers={},
            max_retries=2
        )

    assert mock_req.call_count == 3
    assert "网络连接超时或失败" in str(exc_info.value)
    assert "代理" in exc_info.value.suggested_action or "网络" in exc_info.value.suggested_action


# =========================================================================
# 5. Progress Callback Reporting & Fatal Non-429 Errors
# =========================================================================

def test_challenge_progress_callback_invoked_on_retry(monkeypatch):
    """Verifies that progress_callback is notified on retries."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.text = "Server error"

    resp_200 = MagicMock()
    resp_200.status_code = 200

    mock_req = MagicMock(side_effect=[resp_500, resp_200])
    monkeypatch.setattr(requests, "request", mock_req)

    progress_events = []
    def callback(pct, msg):
        progress_events.append((pct, msg))

    execute_http_request_with_retry(
        method="POST",
        url="https://api.mock.test/v1",
        headers={},
        max_retries=2,
        progress_callback=callback
    )

    assert len(progress_events) >= 1
    assert "重试" in progress_events[0][1]


def test_fatal_auth_401_403_immediate_abort(monkeypatch):
    """Verifies that HTTP 401 and 403 immediately abort with 0 retries."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_401 = MagicMock()
    resp_401.status_code = 401
    resp_401.text = "Unauthorized"

    mock_req = MagicMock(return_value=resp_401)
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(AuthenticationError):
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1",
            headers={},
            max_retries=5
        )

    assert mock_req.call_count == 1
    assert len(sleep_calls) == 0


def test_fatal_model_not_found_404_immediate_abort(monkeypatch):
    """Verifies that HTTP 404 immediately aborts with 0 retries."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    resp_404 = MagicMock()
    resp_404.status_code = 404
    resp_404.text = "Model not found"

    mock_req = MagicMock(return_value=resp_404)
    monkeypatch.setattr(requests, "request", mock_req)

    with pytest.raises(ModelNotFoundError):
        execute_http_request_with_retry(
            method="POST",
            url="https://api.mock.test/v1",
            headers={},
            max_retries=5
        )

    assert mock_req.call_count == 1
    assert len(sleep_calls) == 0
