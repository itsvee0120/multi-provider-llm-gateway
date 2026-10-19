"""Streamlit UI for UnifiedLLMGateway.

Run from the project root:
    uv run streamlit run src/app.py

The gateway is created inside each run (`async with UnifiedLLMGateway() as gw`) and driven with
`asyncio.run`, never cached: its HTTP clients belong to the event loop that created them, and a
cached gateway would outlive that loop ("Event loop is closed").
"""

from __future__ import annotations

import asyncio
import logging
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

# `streamlit run src/app.py` puts src/ (not the project root) on sys.path.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.exceptions import (  # noqa: E402
    AuthenticationError,
    GatewayError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    UnsupportedModelError,
)
from src.gateway import UnifiedLLMGateway  # noqa: E402
from src.pricing import PRICING, ModelPricing  # noqa: E402
from src.schemas import ChunkTelemetry, Provider  # noqa: E402

logger = logging.getLogger(__name__)

PROVIDER_LABELS = {Provider.OPENAI: "OpenAI", Provider.ANTHROPIC: "Anthropic", Provider.GROQ: "Groq"}
TEMPERATURE_IGNORED = "Ignored by this model"
DEFAULT_MAX_TOKENS = 1024


# --------------------------------------------------------------------------- formatting helpers


def models_for_provider(provider: Provider) -> list[str]:
    return sorted(model for model, pricing in PRICING.items() if pricing.provider is provider)


def format_cost(cost_usd: float) -> str:
    """Exactly six decimals, e.g. $0.000142. Rounds half up via Decimal, not binary float."""
    return f"${Decimal(str(cost_usd)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)}"


def format_ttft(ttft_ms: Optional[float]) -> str:
    return "n/a" if ttft_ms is None else f"{ttft_ms:,.0f} ms"


def format_tokens_per_second(tokens_per_second: float) -> str:
    return f"{tokens_per_second:,.1f}"


def prompt_tokens_label(telemetry: ChunkTelemetry) -> str:
    return "Prompt tokens (estimated)" if telemetry.usage_estimated else "Prompt tokens"


def completion_tokens_label(pricing: ModelPricing, telemetry: ChunkTelemetry) -> str:
    notes = []
    if pricing.reasoning:
        notes.append("includes reasoning tokens")
    if telemetry.usage_estimated:
        notes.append("estimated")
    return "Completion tokens" + (f" ({', '.join(notes)})" if notes else "")


def error_message(exc: BaseException) -> str:
    """A short, user-facing message. Never includes the exception's own text, which can come from
    the provider; only the type, provider and safe fields are used."""
    provider = PROVIDER_LABELS.get(getattr(exc, "provider", None), "The provider")
    if isinstance(exc, AuthenticationError):
        return f"{provider} rejected the API key, or it is missing or malformed. Check the key and try again."
    if isinstance(exc, RateLimitExceededError):
        wait = f" Try again in {exc.retry_after_s:g} s." if exc.retry_after_s else " Try again shortly."
        return f"{provider} rate limit reached.{wait}"
    if isinstance(exc, ProviderConnectionError):
        return f"Could not reach {provider}, or the request timed out."
    if isinstance(exc, UnsupportedModelError):
        return "That model is not supported."
    if isinstance(exc, GatewayProviderError):
        status = f" (HTTP {exc.status_code})" if exc.status_code else ""
        return f"{provider} returned an error{status}."
    if isinstance(exc, GatewayError):
        return "The request failed."
    if type(exc) is ValueError:
        # Only the gateway's own input checks (empty prompt, temperature or max_tokens range).
        # Subclasses such as pydantic's ValidationError can echo input data, so they are excluded.
        return str(exc)
    return "Something went wrong. The error type was logged on the server."


# --------------------------------------------------------------------------- UI


async def _run(prompt: str, model: str, temperature: float, max_tokens: int, output) -> Optional[ChunkTelemetry]:
    text = ""
    async with UnifiedLLMGateway() as gw:
        async for chunk in gw.stream(prompt, model, temperature, max_tokens=max_tokens):
            if chunk.is_final:
                output.markdown(text or "_(no visible text)_")  # drop the cursor
                return chunk.telemetry
            text += chunk.delta_text
            output.markdown(text + "▌")
    return None


def _show_telemetry(st, pricing: ModelPricing, t: ChunkTelemetry) -> None:
    st.subheader("Telemetry")
    cols = st.columns(5)
    cols[0].metric("TTFT", format_ttft(t.ttft_ms))
    cols[1].metric("Tokens/sec", format_tokens_per_second(t.tokens_per_second))
    cols[2].metric(prompt_tokens_label(t), f"{t.input_tokens:,}")
    cols[3].metric(completion_tokens_label(pricing, t), f"{t.output_tokens:,}")
    cols[4].metric("Cost", format_cost(t.cost_usd))


def main() -> None:
    import streamlit as st
    from dotenv import load_dotenv

    load_dotenv()  # keys come from the environment; the UI never displays them
    st.set_page_config(page_title="LLM Gateway")
    st.title("Multi-provider LLM gateway")

    provider = st.selectbox("Provider", list(PROVIDER_LABELS), format_func=PROVIDER_LABELS.get)
    model = st.selectbox("Model", models_for_provider(provider))
    pricing = PRICING[model]
    prompt = st.text_area("Prompt", height=150)
    max_tokens = st.number_input("Max tokens", min_value=1, max_value=128_000, value=DEFAULT_MAX_TOKENS, step=1)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        disabled=not pricing.supports_temperature,
        help=None if pricing.supports_temperature else TEMPERATURE_IGNORED,
    )
    if not pricing.supports_temperature:
        st.caption(TEMPERATURE_IGNORED)

    if not st.button("Run", type="primary", disabled=not prompt.strip()):
        return

    output = st.empty()
    try:
        telemetry = asyncio.run(_run(prompt, model, temperature, int(max_tokens), output))
    except Exception as exc:  # never show raw exceptions or tracebacks in the page
        if not isinstance(exc, (GatewayError, ValueError)):
            # Type only: an unexpected exception's message or traceback could hold request data.
            logger.error("Unexpected %s while streaming from %s", type(exc).__name__, model)
        st.error(error_message(exc))
        return

    if telemetry is not None:
        _show_telemetry(st, pricing, telemetry)


if __name__ == "__main__":  # `streamlit run` executes the script as __main__
    main()
