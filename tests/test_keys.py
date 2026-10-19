"""Missing or malformed API keys (assignment Test 4) and key leakage (Test 5).

No live calls: every client points at a closed local port (127.0.0.1:9) with max_retries=0,
and every key is fake.
"""

from __future__ import annotations

import logging

import pytest

from src import gateway as gw
from src.exceptions import AuthenticationError, GatewayError, GatewayProviderError, ProviderConnectionError
from src.gateway import UnifiedLLMGateway

from .fakes import fake_openai

CLOSED_PORT = "http://127.0.0.1:9"  # nothing listens here; no request can leave the machine

PROVIDERS = {
    # provider: (constructor kwarg for the key, base-URL kwarg, model)
    "openai": ("openai_api_key", "openai_base_url", "gpt-5.4"),
    "anthropic": ("anthropic_api_key", "anthropic_base_url", "claude-haiku-4-5"),
    "groq": ("groq_api_key", "groq_base_url", "openai/gpt-oss-20b"),
}

# Fragments of the fake keys that must never appear anywhere in an error or log.
KEY_FRAGMENTS = ("FAKE", "sk-", "É", "\\xc9")

MALFORMED = {
    "empty": "",
    "only-spaces": "   ",
    "inner-space": "sk-FAKE KEY",
    "non-ascii": "sk-FAKÉKEY",
}


@pytest.fixture(autouse=True)
def fixed_tiktoken(monkeypatch):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 1)


def make_gateway(provider: str, key: str) -> UnifiedLLMGateway:
    key_kwarg, url_kwarg, _ = PROVIDERS[provider]
    return UnifiedLLMGateway(**{key_kwarg: key, url_kwarg: CLOSED_PORT}, max_retries=0, timeout_s=2)


async def stream_error(provider: str, key: str, caplog) -> GatewayError:
    gateway = make_gateway(provider, key)
    with caplog.at_level(logging.DEBUG), pytest.raises(GatewayError) as info:
        async for _ in gateway.stream("hi", PROVIDERS[provider][2]):
            pass
    await gateway.aclose()
    return info.value


def assert_no_key_leak(exc: BaseException, caplog) -> None:
    attributes = repr(vars(exc)) + repr(exc.args)
    for text in (str(exc), repr(exc), caplog.text, attributes):
        for fragment in KEY_FRAGMENTS:
            assert fragment not in text, f"key fragment {fragment!r} leaked"
    assert exc.__cause__ is None and exc.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case", MALFORMED)
async def test_malformed_key_raises_authentication_error(provider, case, caplog):
    exc = await stream_error(provider, MALFORMED[case], caplog)

    assert type(exc) is AuthenticationError
    assert exc.provider == provider
    assert "API key is empty or contains invalid characters" in str(exc)
    assert_no_key_leak(exc, caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_trailing_newline_is_stripped_then_request_is_attempted(provider, caplog):
    # Accepted after stripping, so the request really is sent, to a closed port.
    exc = await stream_error(provider, "sk-FAKEKEY\n", caplog)

    assert type(exc) is ProviderConnectionError
    assert exc.provider == provider
    assert_no_key_leak(exc, caplog)


def test_bad_key_does_not_stop_the_gateway_starting():
    gateway = UnifiedLLMGateway(
        openai_api_key="sk-FAKE KEY", anthropic_api_key="", groq_api_key="sk-FAKÉKEY"
    )
    assert "FAKE" not in repr(gateway)


def test_key_is_stored_stripped():
    gateway = UnifiedLLMGateway(openai_api_key="  sk-FAKEKEY\n")
    assert gateway._openai_key.get_secret_value() == "sk-FAKEKEY"


# --------------------------------------------------------------------------- safety net


@pytest.mark.asyncio
async def test_unicode_error_while_building_request_is_typed_and_scrubbed(caplog):
    # What httpx raises for a non-ASCII header: .object is the whole header value, key included.
    header_error = UnicodeEncodeError("ascii", "Bearer sk-FAKÉKEY", 13, 14, "ordinal not in range(128)")
    gateway = UnifiedLLMGateway(openai_client=fake_openai(header_error))

    with caplog.at_level(logging.DEBUG), pytest.raises(GatewayProviderError) as info:
        async for _ in gateway.stream("hi", "gpt-5.4"):
            pass

    assert "UnicodeEncodeError" in str(info.value)
    assert_no_key_leak(info.value, caplog)


@pytest.mark.asyncio
async def test_unexpected_error_message_is_redacted(caplog):
    gateway = UnifiedLLMGateway(openai_client=fake_openai(RuntimeError("boom with sk-FAKEKEY1234567890")))

    with caplog.at_level(logging.DEBUG), pytest.raises(GatewayProviderError) as info:
        async for _ in gateway.stream("hi", "gpt-5.4"):
            pass

    assert "RuntimeError" in str(info.value) and "[REDACTED]" in str(info.value)
    assert_no_key_leak(info.value, caplog)
