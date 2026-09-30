"""Provider error classification against error bodies captured from the real
APIs during this project (not invented). Where a body was only partly
captured, the fixture says which part was reconstructed."""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime

import httpx
import pytest

from app.observability.logging_setup import ContextFilter, JsonFormatter
from app.observability.provider_errors import (
    ErrorKind,
    classify_exception,
    classify_google,
    classify_http,
    find_failure,
    log_provider_failure,
)
from app.orchestration.model_provider import ModelProviderError
from app.services.embedding import EmbeddingProviderError

# --- Gemini -----------------------------------------------------------------

# 429 on gemini-2.5-flash chart extraction, 2026-09-28. The message is verbatim
# as captured; the capture was cut off inside `details` after the Help entry,
# so the QuotaFailure/RetryInfo entries below follow Google's documented
# google.rpc shape and are reconstructed, not captured.
GEMINI_429_MESSAGE = (
    "You exceeded your current quota, please check your plan and billing details. For more "
    "information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To "
    "monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for "
    "metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
    "model: gemini-2.5-flash\nPlease retry in 43.342838031s."
)
GEMINI_429_DETAILS = {
    "error": {
        "code": 429,
        "message": GEMINI_429_MESSAGE,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests",
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaValue": "20",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"},
        ],
    }
}

# 503 via ADK's error event (error_code/error_message), 2026-09-29. Verbatim.
GEMINI_503_MESSAGE = (
    "This model is currently experiencing high demand. Spikes in demand are usually "
    "temporary. Please try again later."
)

# 400 invalid key via google-genai ClientError, captured 2026-09-30. Verbatim.
GEMINI_INVALID_KEY_DETAILS = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"},
            },
            {
                "@type": "type.googleapis.com/google.rpc.LocalizedMessage",
                "locale": "en-US",
                "message": "API key not valid. Please pass a valid API key.",
            },
        ],
    }
}

# --- OpenRouter ---------------------------------------------------------------

# 429 on /api/v1/embeddings once the free-model daily cap was used up,
# captured 2026-09-30. Verbatim.
OPENROUTER_429_DAILY = {
    "error": {
        "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 "
        "free model requests per day",
        "code": 429,
        "metadata": {
            "headers": {
                "X-RateLimit-Limit": "50",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1790726400000",
            },
            "limit_source": "openrouter_free_tier_daily",
            "remedy_hint": "Wait for the daily reset (see X-RateLimit-Reset), or purchase "
            "credits to raise your free-model daily limit.",
        },
    }
}

# HTTP 200 whose body is an upstream error, 2026-09-28. Verbatim.
OPENROUTER_200_EMBEDDED = {
    "id": "gen-1790599867-Ezfxmwsg8xyrX0ZUtynZ",
    "error": {
        "message": "Upstream error from Nvidia: Service temporarily overloaded",
        "code": 503,
        "metadata": {"error_type": "provider_overloaded"},
    },
}

# 401 for an unknown key, captured 2026-09-30. Verbatim.
OPENROUTER_401 = {"error": {"message": "User not found.", "code": 401}}

# 502 on /api/v1/embeddings, 2026-09-28 (message as surfaced by the provider).
OPENROUTER_502 = {"error": {"message": "HTTP 502: error code: 502"}}

# --- xAI ----------------------------------------------------------------------

# 400 for a non-xAI key (a Groq gsk_ key), captured 2026-09-30. Verbatim.
XAI_INVALID_KEY = {
    "code": "invalid-argument",
    "error": "Incorrect API key provided. You can obtain an API key from https://console.x.ai.",
}


def test_gemini_429_is_quota_exceeded_with_metric_limit_window_and_retry():
    failure = classify_google(
        code=429,
        status="RESOURCE_EXHAUSTED",
        message=GEMINI_429_MESSAGE,
        details=GEMINI_429_DETAILS,
        operation="chart",
        model=None,
    )
    assert failure.kind is ErrorKind.QUOTA_EXCEEDED
    assert failure.model == "gemini-2.5-flash"
    assert failure.quota == "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    assert failure.limit == 20
    assert failure.window == "per-day"
    assert failure.retry_after_s == pytest.approx(43.34, abs=0.01)
    assert failure.summary().startswith("[QUOTA_EXCEEDED] gemini/gemini-2.5-flash chart:")


def test_gemini_429_from_the_captured_message_text_alone():
    # What ADK's error event carries: the status string and message, no details.
    failure = classify_google(
        code=None,
        status="RESOURCE_EXHAUSTED",
        message=GEMINI_429_MESSAGE,
        operation="chat",
        model="gemini-2.5-flash",
    )
    assert failure.kind is ErrorKind.QUOTA_EXCEEDED
    assert failure.limit == 20
    assert failure.window is None  # the text alone doesn't say per-day vs per-minute
    assert failure.retry_after_s == pytest.approx(43.34, abs=0.01)


def test_gemini_per_minute_quota_is_rate_limited():
    details = json.loads(json.dumps(GEMINI_429_DETAILS))
    details["error"]["details"][1]["violations"][0]["quotaId"] = (
        "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    )
    failure = classify_google(
        code=429,
        status="RESOURCE_EXHAUSTED",
        message=GEMINI_429_MESSAGE,
        details=details,
        operation="chat",
        model="gemini-2.5-flash",
    )
    assert failure.kind is ErrorKind.RATE_LIMITED
    assert failure.window == "per-minute"


def test_gemini_503_from_adk_error_event_is_overloaded():
    failure = classify_google(
        code=None,
        status="UNAVAILABLE",
        message=GEMINI_503_MESSAGE,
        operation="chat",
        model="gemini-2.5-flash",
    )
    assert failure.kind is ErrorKind.OVERLOADED
    assert "high demand" in failure.message


def test_gemini_invalid_key_is_recognised_from_error_info_reason():
    error = GEMINI_INVALID_KEY_DETAILS["error"]
    failure = classify_google(
        code=error["code"],
        status=error["status"],
        message=error["message"],
        details=GEMINI_INVALID_KEY_DETAILS,
        operation="chat",
        model="gemini-2.5-flash",
    )
    assert failure.kind is ErrorKind.INVALID_KEY
    assert "GEMINI_API_KEY" in failure.summary()


def test_genai_api_error_exception_is_classified_through_the_exception_chain():
    from google.genai import errors as genai_errors

    api_error = genai_errors.ClientError(429, GEMINI_429_DETAILS)
    try:
        try:
            raise api_error
        except Exception as exc:
            raise ModelProviderError("Gemini request failed") from exc
    except ModelProviderError as wrapped:
        failure = classify_exception(
            wrapped, provider="gemini", operation="chat", model="gemini-2.5-flash"
        )
    assert failure.kind is ErrorKind.QUOTA_EXCEEDED
    assert failure.limit == 20


def test_openrouter_daily_cap_is_quota_exceeded_with_limit_and_reset_time():
    failure = classify_http(
        provider="openrouter",
        status=429,
        body=OPENROUTER_429_DAILY,
        operation="embedding",
        model="nvidia/nemotron-3-embed-1b:free",
    )
    assert failure.kind is ErrorKind.QUOTA_EXCEEDED
    assert failure.quota == "free-models-per-day"
    assert failure.limit == 50
    assert failure.window == "per-day"
    # 1790726400000 ms is 2026-09-30 00:00 UTC, the next UTC midnight when captured.
    assert failure.resets_at == datetime(2026, 9, 30, 0, 0, tzinfo=UTC)
    assert failure.summary() == (
        "[QUOTA_EXCEEDED] openrouter/nvidia/nemotron-3-embed-1b:free embedding: "
        "quota free-models-per-day; limit 50 per-day; resets 2026-09-30 00:00 UTC"
    )


def test_openrouter_short_window_429_is_rate_limited():
    body = {"error": {"message": "Rate limit exceeded: free-models-per-min.", "code": 429}}
    failure = classify_http(
        provider="openrouter", status=429, body=body, operation="chat", model="m"
    )
    assert failure.kind is ErrorKind.RATE_LIMITED
    assert failure.quota == "free-models-per-min"


def test_openrouter_200_with_embedded_upstream_error_is_overloaded():
    failure = classify_http(
        provider="openrouter",
        status=200,
        body=OPENROUTER_200_EMBEDDED,
        operation="chat",
        model="nvidia/nemotron-3-super-120b-a12b:free",
    )
    assert failure.kind is ErrorKind.OVERLOADED
    assert failure.status == 503  # the embedded code, not the misleading HTTP 200


def test_openrouter_unknown_key_is_invalid_key():
    failure = classify_http(
        provider="openrouter", status=401, body=OPENROUTER_401, operation="chat", model="m"
    )
    assert failure.kind is ErrorKind.INVALID_KEY
    assert "OPENROUTER_API_KEY" in failure.summary()


def test_openrouter_502_is_upstream_error():
    failure = classify_http(
        provider="openrouter", status=502, body=OPENROUTER_502, operation="embedding", model="m"
    )
    assert failure.kind is ErrorKind.UPSTREAM_ERROR


def test_xai_400_with_wrong_key_is_invalid_key_not_bad_request():
    failure = classify_http(
        provider="xai", status=400, body=XAI_INVALID_KEY, operation="chat", model="grok-4.3"
    )
    assert failure.kind is ErrorKind.INVALID_KEY
    assert "XAI_API_KEY" in failure.summary()


def test_transport_failure_is_network():
    request = httpx.Request("POST", "https://example.invalid")
    failure = classify_exception(
        httpx.ConnectError("connection refused", request=request),
        provider="openrouter",
        operation="chat",
        model="m",
    )
    assert failure.kind is ErrorKind.NETWORK


def test_failure_is_found_through_a_wrapping_exception():
    failure = classify_http(
        provider="openrouter",
        status=429,
        body=OPENROUTER_429_DAILY,
        operation="embedding",
        model="m",
    )
    try:
        try:
            raise EmbeddingProviderError("OpenRouter returned an error", failure=failure)
        except EmbeddingProviderError as exc:
            raise RuntimeError("retrieval backend is unavailable") from exc
    except RuntimeError as outer:
        assert find_failure(outer) is failure


def test_a_provider_failure_logs_one_json_line_without_a_traceback_at_info():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    test_logger = logging.getLogger("test.provider_failure")
    test_logger.handlers, test_logger.propagate = [handler], False
    test_logger.setLevel(logging.INFO)
    failure = classify_http(
        provider="openrouter",
        status=429,
        body=OPENROUTER_429_DAILY,
        operation="embedding",
        model="nvidia/nemotron-3-embed-1b:free",
    )

    try:
        raise EmbeddingProviderError("OpenRouter returned an error (status=429)", failure=failure)
    except EmbeddingProviderError as exc:
        log_provider_failure(test_logger, failure, exc)

    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["level"] == "WARNING"
    assert record["msg"].startswith("[QUOTA_EXCEEDED] openrouter/")
    assert record["event"] == "provider_failure"
    assert record["failure"]["kind"] == "QUOTA_EXCEEDED"
    assert record["failure"]["limit"] == 50
    assert "exc" not in record  # traceback only at DEBUG


def test_the_traceback_is_still_available_at_debug():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    test_logger = logging.getLogger("test.provider_failure_debug")
    test_logger.handlers, test_logger.propagate = [handler], False
    test_logger.setLevel(logging.DEBUG)
    failure = classify_http(
        provider="xai", status=400, body=XAI_INVALID_KEY, operation="chat", model="grok-4.3"
    )
    try:
        raise ModelProviderError("xAI returned an error (status=400)", failure=failure)
    except ModelProviderError as exc:
        log_provider_failure(test_logger, failure, exc)

    records = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    assert [r["level"] for r in records] == ["WARNING", "DEBUG"]
    assert "Traceback" in records[1]["exc"]
