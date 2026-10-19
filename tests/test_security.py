from __future__ import annotations

import logging

import anthropic
import openai
import pytest

from src import gateway as gw
from src.gateway import (
    AuthenticationError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    UnifiedLLMGateway,
    redact_secrets,
)

from .fakes import FakeStream, fake_anthropic, fake_openai, http_response, openai_chunk

OPENAI_KEY = "sk-proj-TESTKEY1234567890abcdef"
ANTHROPIC_KEY = "sk-ant-api03-TESTKEY1234567890abcdef"
GROQ_KEY = "gsk_TESTKEY1234567890abcdef"


async def collect(gen):
    return [chunk async for chunk in gen]


@pytest.fixture(autouse=True)
def fixed_tiktoken(monkeypatch):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 1)


def assert_no_secret(exc: BaseException, caplog) -> None:
    for text in (str(exc), repr(exc), caplog.text):
        assert OPENAI_KEY not in text and ANTHROPIC_KEY not in text and GROQ_KEY not in text
        assert "TESTKEY" not in text
    assert exc.__context__ is None and exc.__cause__ is None


@pytest.mark.asyncio
async def test_invalid_openai_key_maps_to_auth_error_without_leaking(caplog):
    sdk_error = openai.AuthenticationError(
        f"Incorrect API key provided: {OPENAI_KEY}", response=http_response(401), body=None
    )
    gateway = UnifiedLLMGateway(openai_api_key=OPENAI_KEY, openai_client=fake_openai(sdk_error))

    with caplog.at_level(logging.WARNING), pytest.raises(AuthenticationError) as info:
        await collect(gateway.stream("hi", "gpt-5.4"))

    assert info.value.status_code == 401
    assert "[REDACTED]" in str(info.value)
    assert_no_secret(info.value, caplog)


@pytest.mark.asyncio
async def test_rate_limit_carries_retry_after(caplog):
    sdk_error = anthropic.RateLimitError(
        "rate limited", response=http_response(429, {"retry-after": "7"}), body=None
    )
    gateway = UnifiedLLMGateway(anthropic_api_key=ANTHROPIC_KEY, anthropic_client=fake_anthropic(sdk_error))

    with caplog.at_level(logging.WARNING), pytest.raises(RateLimitExceededError) as info:
        await collect(gateway.stream("hi", "claude-opus-5-5"))

    assert info.value.retry_after_s == 7.0
    assert isinstance(info.value, GatewayProviderError)
    assert_no_secret(info.value, caplog)


@pytest.mark.asyncio
async def test_connection_error_mid_stream_is_typed(caplog):
    request = http_response(200).request
    stream = FakeStream([openai_chunk("partial")], fail_with=openai.APITimeoutError(request=request))
    gateway = UnifiedLLMGateway(openai_api_key=OPENAI_KEY, openai_client=fake_openai(stream))

    received = []
    with caplog.at_level(logging.WARNING), pytest.raises(ProviderConnectionError) as info:
        async for chunk in gateway.stream("hi", "gpt-5.4"):
            received.append(chunk.delta_text)

    assert received == ["partial"]
    assert stream.closed
    assert_no_secret(info.value, caplog)


@pytest.mark.asyncio
async def test_missing_anthropic_key_is_an_auth_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AuthenticationError):
        await collect(UnifiedLLMGateway().stream("hi", "claude-haiku-4-5"))


@pytest.mark.asyncio
async def test_missing_openai_key_is_an_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AuthenticationError):
        await collect(UnifiedLLMGateway().stream("hi", "gpt-5.4"))


@pytest.mark.asyncio
async def test_invalid_groq_key_maps_to_auth_error_without_leaking(caplog):
    sdk_error = openai.AuthenticationError(f"Invalid API Key: {GROQ_KEY}", response=http_response(401), body=None)
    gateway = UnifiedLLMGateway(groq_api_key=GROQ_KEY, groq_client=fake_openai(sdk_error))

    with caplog.at_level(logging.WARNING), pytest.raises(AuthenticationError) as info:
        await collect(gateway.stream("hi", "openai/gpt-oss-120b"))

    assert info.value.provider == "groq" and info.value.status_code == 401
    assert "[REDACTED]" in str(info.value)
    assert_no_secret(info.value, caplog)


@pytest.mark.asyncio
async def test_groq_rate_limit_is_typed():
    sdk_error = openai.RateLimitError("slow down", response=http_response(429, {"retry-after": "2"}), body=None)
    gateway = UnifiedLLMGateway(groq_client=fake_openai(sdk_error))
    with pytest.raises(RateLimitExceededError) as info:
        await collect(gateway.stream("hi", "openai/gpt-oss-20b"))
    assert info.value.retry_after_s == 2.0


def test_gateway_repr_hides_keys():
    gateway = UnifiedLLMGateway(openai_api_key=OPENAI_KEY, anthropic_api_key=ANTHROPIC_KEY, groq_api_key=GROQ_KEY)
    assert "TESTKEY" not in repr(gateway)
    assert "TESTKEY" not in repr(vars(gateway))


@pytest.mark.parametrize(
    "text",
    [
        f"key={OPENAI_KEY}",
        f"Authorization: Bearer {OPENAI_KEY}",
        f"x-api-key: {ANTHROPIC_KEY}",
        "Incorrect API key provided: sk-proj-********cdef",
        f"Invalid API Key: {GROQ_KEY}",
        "Invalid API Key: gsk_****************cdef",
    ],
)
def test_redact_secrets_patterns(text):
    redacted = redact_secrets(text)
    assert "[REDACTED]" in redacted
    assert "TESTKEY" not in redacted and "cdef" not in redacted
