"""Async streaming gateway that presents OpenAI and Anthropic behind one interface."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Callable, Iterable, NamedTuple, Optional

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr

from .exceptions import (
    AuthenticationError,
    GatewayError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    UnsupportedModelError,
)
from .pricing import ModelPricing, require_pricing
from .schemas import ChunkTelemetry, Provider, StreamChunk

__all__ = [
    "AuthenticationError",
    "GatewayError",
    "GatewayProviderError",
    "ProviderConnectionError",
    "RateLimitExceededError",
    "UnifiedLLMGateway",
    "UnsupportedModelError",
    "redact_secrets",
]

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
# tiktoken counts the raw prompt; the API also counts chat-template tokens around it.
PROMPT_TOKEN_TOLERANCE = 32
FALLBACK_ENCODING = "o200k_base"

OPENAI_TEMPERATURE_RANGE = (0.0, 2.0)

# Credential shapes that must never reach logs or exception messages, even partially masked.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-*]{6,}"),
    re.compile(r"gsk_[A-Za-z0-9_\-*]{6,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]+"),
)
REDACTED = "[REDACTED]"


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + REDACTED, text)
    return text


# --------------------------------------------------------------------------- telemetry


class _StreamMetrics:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._first_token_at: Optional[float] = None

    def mark_token(self) -> None:
        if self._first_token_at is None:
            self._first_token_at = time.perf_counter()

    def build(
        self,
        pricing: ModelPricing,
        input_tokens: int,
        output_tokens: int,
        estimated_input_tokens: Optional[int] = None,
        *,
        usage_estimated: bool = False,
        reasoning_tokens: Optional[int] = None,
    ) -> ChunkTelemetry:
        latency_s = time.perf_counter() - self._start
        ttft_ms = None if self._first_token_at is None else (self._first_token_at - self._start) * 1000
        return ChunkTelemetry(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_latency_s=latency_s,
            ttft_ms=ttft_ms,
            tokens_per_second=output_tokens / latency_s if latency_s > 0 else 0.0,
            cost_usd=pricing.cost_usd(input_tokens, output_tokens),
            estimated_input_tokens=estimated_input_tokens,
            usage_estimated=usage_estimated,
            reasoning_tokens=reasoning_tokens,
        )


@functools.lru_cache(maxsize=32)
def _encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def count_tokens(text: str, model: str) -> Optional[int]:
    """Count tokens locally with tiktoken; None if no encoding could be loaded (e.g. offline)."""
    try:
        return len(_encoding_for(model).encode(text))
    except Exception:  # tiktoken may need to download its BPE file on first use
        logger.warning("tiktoken unavailable for %s; skipping local token count", model, exc_info=True)
        return None


# --------------------------------------------------------------------------- gateway

_OPENAI_SDK_ERRORS = (openai.OpenAIError,)
_ANTHROPIC_SDK_ERRORS = (anthropic.AnthropicError,)


class UnifiedLLMGateway:
    """Stream completions from OpenAI, Anthropic or Groq as provider-agnostic `StreamChunk`s.

    API keys default to OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY and are stripped of
    surrounding whitespace. A missing or malformed key only fails when a model from that provider
    is requested, so one bad key never stops the gateway starting. The OpenAI and Anthropic
    clients are created on first use; the Groq client is created here when GROQ_API_KEY is set.
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        openai_client: Optional[AsyncOpenAI] = None,
        anthropic_client: Optional[AsyncAnthropic] = None,
        groq_api_key: Optional[str] = None,
        groq_client: Optional[AsyncOpenAI] = None,
        openai_base_url: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
        groq_base_url: str = GROQ_BASE_URL,
    ) -> None:
        self._openai_key = _secret(_resolve_key(openai_api_key, "OPENAI_API_KEY"))
        self._anthropic_key = _secret(_resolve_key(anthropic_api_key, "ANTHROPIC_API_KEY"))
        self._groq_key = _secret(_resolve_key(groq_api_key, "GROQ_API_KEY"))
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._openai_base_url = openai_base_url
        self._anthropic_base_url = anthropic_base_url
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client
        # Groq is OpenAI-compatible: same SDK, different base URL. Created once, here; without a
        # key it stays None and _groq() raises AuthenticationError when a Groq model is requested.
        # Building the client does not validate or send the key; _groq() validates it first.
        self._groq_client = groq_client
        if self._groq_client is None and self._groq_key is not None and self._groq_key.get_secret_value():
            self._groq_client = AsyncOpenAI(
                api_key=self._groq_key.get_secret_value(),
                base_url=groq_base_url,
                timeout=timeout_s,
                max_retries=max_retries,
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(openai_configured={self._openai_key is not None}, "
            f"anthropic_configured={self._anthropic_key is not None}, "
            f"groq_configured={self._groq_key is not None})"
        )

    async def __aenter__(self) -> UnifiedLLMGateway:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for client in (self._openai_client, self._anthropic_client, self._groq_client):
            if client is not None:
                await client.close()

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        *,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield text deltas, then one final chunk (`is_final=True`, empty text) carrying telemetry.

        `temperature` is only sent to models that accept it (see
        `ModelPricing.supports_temperature`); other models use their default sampling.
        `max_tokens` defaults to the provider's own limit for OpenAI and Groq, and to
        DEFAULT_ANTHROPIC_MAX_TOKENS for Anthropic, where it is mandatory.
        """
        pricing = require_pricing(model)
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        if pricing.supports_temperature:
            low, high = OPENAI_TEMPERATURE_RANGE
            if not low <= temperature <= high:
                raise ValueError(f"temperature must be between {low} and {high}")
        else:
            logger.debug("%s does not accept temperature; using the model's default sampling", model)
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if pricing.provider is Provider.OPENAI:
            source = self._stream_openai_compatible(
                Provider.OPENAI, self._openai, prompt, model, temperature, max_tokens, pricing
            )
        elif pricing.provider is Provider.GROQ:
            source = self._stream_openai_compatible(
                Provider.GROQ, self._groq, prompt, model, temperature, max_tokens, pricing
            )
        else:
            source = self._stream_anthropic(prompt, model, max_tokens, pricing)
        async with contextlib.aclosing(source):
            async for chunk in source:
                yield chunk

    # ----------------------------------------------------------------------- OpenAI & Groq

    async def _stream_openai_compatible(
        self,
        provider: Provider,
        get_client: Callable[[], AsyncOpenAI],
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        pricing: ModelPricing,
    ) -> AsyncGenerator[StreamChunk, None]:
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if pricing.supports_temperature:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens

        # Off the event loop: encoding a large prompt is CPU-bound. Done before the timer starts.
        estimated_input = await asyncio.to_thread(count_tokens, prompt, model)
        metrics = _StreamMetrics()
        usage = None
        text_parts: list[str] = []
        error: Optional[GatewayProviderError] = None
        try:
            stream = await get_client().chat.completions.create(**params)
            async with stream:
                async for event in stream:
                    usage = _usage_tokens(event) or usage  # usually on a trailing chunk with no choices
                    for choice in event.choices or ():
                        text = choice.delta.content if choice.delta else None
                        if choice.index == 0 and text:
                            metrics.mark_token()
                            text_parts.append(text)
                            yield StreamChunk(delta_text=text, provider=provider, model=model)
        except _OPENAI_SDK_ERRORS as exc:
            error = self._map_error(provider, exc)
        except GatewayProviderError as exc:  # e.g. a malformed key, from get_client()
            error = exc
        except Exception as exc:
            error = self._map_unexpected(provider, exc)
        if error is not None:
            raise error  # raised outside `except` so the original exception is not attached as context

        if usage is not None:
            input_tokens, output_tokens = usage.prompt, usage.completion
            self._verify_prompt_tokens(model, input_tokens, estimated_input, pricing.tiktoken_comparable)
        else:
            logger.warning(
                "%s stream for %s ended without usage; falling back to tiktoken counts", provider, model
            )
            input_tokens = estimated_input or 0
            output_tokens = await asyncio.to_thread(count_tokens, "".join(text_parts), model) or 0

        yield StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=metrics.build(
                pricing,
                input_tokens,
                output_tokens,
                estimated_input,
                usage_estimated=usage is None,
                reasoning_tokens=usage.reasoning if usage is not None else None,
            ),
        )

    @staticmethod
    def _verify_prompt_tokens(model: str, reported: int, estimated: Optional[int], comparable: bool) -> None:
        if estimated is None:
            return
        if not comparable:
            # Known large gap (hidden provider prompt or different tokenizer): diagnostic only.
            logger.debug("Prompt tokens for %s: provider reported %d, tiktoken %d", model, reported, estimated)
            return
        if reported < estimated or reported - estimated > PROMPT_TOKEN_TOLERANCE:
            logger.warning(
                "Prompt token mismatch for %s: provider reported %d, tiktoken estimated %d",
                model,
                reported,
                estimated,
            )

    # ----------------------------------------------------------------------- Anthropic

    async def _stream_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int],
        pricing: ModelPricing,
    ) -> AsyncGenerator[StreamChunk, None]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        metrics = _StreamMetrics()
        input_tokens = output_tokens = 0
        error: Optional[GatewayProviderError] = None
        try:
            stream = await self._anthropic().messages.create(**params)
            async with stream:
                async for event in stream:
                    if event.type == "message_start":
                        input_tokens = event.message.usage.input_tokens
                        output_tokens = event.message.usage.output_tokens
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta" and event.delta.text:
                            metrics.mark_token()
                            yield StreamChunk(
                                delta_text=event.delta.text, provider=Provider.ANTHROPIC, model=model
                            )
                    elif event.type == "message_delta":
                        # Usage on message_delta is cumulative for the message, so the latest
                        # value replaces the running total rather than being added to it.
                        output_tokens = event.usage.output_tokens
                        if event.usage.input_tokens is not None:
                            input_tokens = event.usage.input_tokens
        except _ANTHROPIC_SDK_ERRORS as exc:
            error = self._map_error(Provider.ANTHROPIC, exc)
        except GatewayProviderError as exc:  # e.g. a malformed key, from _anthropic()
            error = exc
        except TypeError as exc:
            if "authentication method" in str(exc):
                # The Anthropic SDK signals "no credentials configured" with a bare TypeError.
                error = AuthenticationError(
                    "No Anthropic credentials configured; set ANTHROPIC_API_KEY", provider=Provider.ANTHROPIC
                )
            else:
                error = self._map_unexpected(Provider.ANTHROPIC, exc)
        except Exception as exc:
            error = self._map_unexpected(Provider.ANTHROPIC, exc)
        if error is not None:
            raise error

        yield StreamChunk(
            delta_text="",
            provider=Provider.ANTHROPIC,
            model=model,
            is_final=True,
            telemetry=metrics.build(pricing, input_tokens, output_tokens),
        )

    # ----------------------------------------------------------------------- clients & errors

    def _openai(self) -> AsyncOpenAI:
        _check_key_format(Provider.OPENAI, self._openai_key)
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=_reveal(self._openai_key),
                base_url=self._openai_base_url,
                timeout=self._timeout_s,
                max_retries=self._max_retries,
            )
        return self._openai_client

    def _anthropic(self) -> AsyncAnthropic:
        _check_key_format(Provider.ANTHROPIC, self._anthropic_key)
        if self._anthropic_client is None:
            self._anthropic_client = AsyncAnthropic(
                api_key=_reveal(self._anthropic_key),
                base_url=self._anthropic_base_url,
                timeout=self._timeout_s,
                max_retries=self._max_retries,
            )
        return self._anthropic_client

    def _groq(self) -> AsyncOpenAI:
        _check_key_format(Provider.GROQ, self._groq_key)
        if self._groq_client is None:
            raise AuthenticationError(
                "No Groq API key configured; set GROQ_API_KEY to use Groq models", provider=Provider.GROQ
            )
        return self._groq_client

    def _map_unexpected(self, provider: Provider, exc: Exception) -> GatewayProviderError:
        """Turn a non-SDK failure while building or sending a request into a typed error.

        The original exception is dropped, not chained: its attributes can hold raw header values
        (UnicodeEncodeError.object is the whole `Bearer <key>` string). Unicode errors keep only their
        type name, because their message quotes the offending character, which may be part of a key.
        """
        name = type(exc).__name__
        if isinstance(exc, UnicodeError):
            detail = "a request header could not be encoded"
        else:
            keys = (self._openai_key, self._anthropic_key, self._groq_key)
            detail = redact_secrets(str(exc), [k.get_secret_value() for k in keys if k is not None])
        mapped = GatewayProviderError(f"Request could not be built or sent ({name}): {detail}", provider=provider)
        logger.warning("%s request failed before a response: %s", provider, mapped)
        return mapped

    def _map_error(self, provider: Provider, exc: Exception) -> GatewayProviderError:
        keys = (self._openai_key, self._anthropic_key, self._groq_key)
        secrets = [k.get_secret_value() for k in keys if k is not None]
        message = redact_secrets(getattr(exc, "message", None) or str(exc) or type(exc).__name__, secrets)
        status_code = getattr(exc, "status_code", None)
        context: dict[str, Any] = {
            "provider": provider,
            "status_code": status_code,
            "request_id": getattr(exc, "request_id", None),
        }

        mapped: GatewayProviderError
        if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
            mapped = RateLimitExceededError(message, retry_after_s=_retry_after(exc), **context)
        elif isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
            mapped = AuthenticationError(message, **context)
        elif isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
            # Timeouts subclass APIConnectionError in both SDKs.
            mapped = ProviderConnectionError(message, **context)
        elif status_code is None and "api_key" in message.lower():
            # Raised by the SDK before any request when no key is configured.
            mapped = AuthenticationError(message, **context)
        else:
            mapped = GatewayProviderError(message, **context)

        logger.warning(
            "%s call failed: %s (status=%s, request_id=%s): %s",
            provider,
            type(mapped).__name__,
            status_code,
            mapped.request_id,
            message,
        )
        return mapped


def _resolve_key(explicit: Optional[str], env_var: str) -> Optional[str]:
    # An explicitly passed key, even "", wins over the environment.
    return explicit if explicit is not None else os.environ.get(env_var)


def _secret(value: Optional[str]) -> Optional[SecretStr]:
    """None means "no key configured"; anything else is kept (stripped) and validated on first use."""
    return SecretStr(value.strip()) if value is not None else None


_PROVIDER_LABELS = {Provider.OPENAI: "OpenAI", Provider.ANTHROPIC: "Anthropic", Provider.GROQ: "Groq"}
# Keys travel in an HTTP header: printable ASCII only, no whitespace or control characters.
# Prefixes (sk-, gsk_, ...) are deliberately not checked.
_VALID_KEY = re.compile(r"[\x21-\x7e]+")


def _check_key_format(provider: Provider, key: Optional[SecretStr]) -> None:
    if key is not None and not _VALID_KEY.fullmatch(key.get_secret_value()):
        # Never include any part of the key in the message.
        raise AuthenticationError(
            f"{_PROVIDER_LABELS[provider]} API key is empty or contains invalid characters", provider=provider
        )


def _reveal(value: Optional[SecretStr]) -> Optional[str]:
    return value.get_secret_value() if value is not None else None


class _Usage(NamedTuple):
    prompt: int
    completion: int
    reasoning: Optional[int]  # subset of `completion`; None when the provider doesn't report it


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _usage_tokens(event: Any) -> Optional[_Usage]:
    """Token usage from an OpenAI-compatible chunk, or None if it carries no usage.

    Groq has also reported usage under a non-standard `x_groq.usage` field, so check there too.
    """
    usage = getattr(event, "usage", None)
    if usage is None:
        x_groq = getattr(event, "x_groq", None)
        usage = x_groq.get("usage") if isinstance(x_groq, dict) else getattr(x_groq, "usage", None)
    if usage is None:
        return None
    details = _field(usage, "completion_tokens_details")
    reasoning = _field(details, "reasoning_tokens") if details is not None else None
    return _Usage(_field(usage, "prompt_tokens"), _field(usage, "completion_tokens"), reasoning)


def _retry_after(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None

