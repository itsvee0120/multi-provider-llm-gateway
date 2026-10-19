"""Static model registry: which provider serves a model and what it costs.

Rates are standard-tier USD per 1,000,000 tokens, taken from the providers' public
pricing pages (OpenAI and Anthropic verified 2026-10-18, Groq 2026-10-19). Batch, priority,
cached-input and data-residency rates are not modelled, so if Anthropic ever reports cache
read/write tokens the cost will be slightly off; the gateway does not use prompt caching.
Re-check these pages before relying on the numbers for billing:
  - https://developers.openai.com/api/docs/pricing
  - https://platform.claude.com/docs/en/about-claude/pricing
  - https://console.groq.com/docs/models
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping, Optional

from .exceptions import UnsupportedModelError
from .schemas import Provider

TOKENS_PER_UNIT = 1_000_000
# Enough precision that a single token on the cheapest model is still non-zero.
# Round to cents or micro-dollars only for display.
COST_DECIMAL_PLACES = 9


def _d(x: float) -> Decimal:
    # Via str() so 0.1 becomes Decimal("0.1"), not the nearest binary float.
    return Decimal(str(x))


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: Provider
    input_per_mtok: float
    output_per_mtok: float
    # The current Anthropic Messages API exposes no sampling parameters.
    supports_temperature: bool = True
    # Optional higher tier applied to the whole request once the prompt exceeds the threshold.
    long_context_threshold: Optional[int] = None
    long_input_per_mtok: Optional[float] = None
    long_output_per_mtok: Optional[float] = None
    # Bills hidden reasoning tokens as output: output_tokens can far exceed the visible reply.
    reasoning: bool = False
    # True when tiktoken's prompt count is a meaningful cross-check of the provider's count.
    # False for Anthropic (different tokenizer) and for models whose provider adds a large
    # hidden prompt overhead (gpt-oss on Groq: 77 tokens billed for a ~6-token prompt).
    tiktoken_comparable: bool = True

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Exact decimal cost, rounded to COST_DECIMAL_PLACES (floats gave e.g. 8.599999999999999e-06)."""
        input_rate, output_rate = _d(self.input_per_mtok), _d(self.output_per_mtok)
        if self.long_context_threshold is not None and input_tokens > self.long_context_threshold:
            # `is not None`, not `or`: a legitimate 0.0 long-tier rate must not fall back to the base rate.
            if self.long_input_per_mtok is not None:
                input_rate = _d(self.long_input_per_mtok)
            if self.long_output_per_mtok is not None:
                output_rate = _d(self.long_output_per_mtok)
        cost = (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / TOKENS_PER_UNIT
        return float(round(cost, COST_DECIMAL_PLACES))


def _openai(input_rate: float, output_rate: float) -> ModelPricing:
    # Every listed OpenAI model is a GPT-5-or-later reasoning model.
    return ModelPricing(Provider.OPENAI, input_rate, output_rate, reasoning=True)


def _anthropic(input_rate: float, output_rate: float, **long_tier: float) -> ModelPricing:
    # No extended thinking is requested, so output tokens are visible text only.
    return ModelPricing(
        Provider.ANTHROPIC,
        input_rate,
        output_rate,
        supports_temperature=False,
        tiktoken_comparable=False,
        **long_tier,
    )


def _groq_gpt_oss(input_rate: float, output_rate: float) -> ModelPricing:
    return ModelPricing(Provider.GROQ, input_rate, output_rate, reasoning=True, tiktoken_comparable=False)


PRICING: Mapping[str, ModelPricing] = MappingProxyType(
    {
        # OpenAI (short-context rates, prompts <= 272K tokens)
        "gpt-6-astra": _openai(10.00, 50.00),
        "gpt-6.1-sol": _openai(2.00, 10.00),
        "gpt-6-sol": _openai(2.00, 10.00),
        "gpt-6-luna": _openai(0.10, 0.50),
        "gpt-5.6-sol": _openai(4.00, 20.00),
        "gpt-5.6-terra": _openai(2.00, 12.00),
        "gpt-5.6-luna": _openai(0.20, 1.20),
        "gpt-5.5": _openai(5.00, 30.00),
        "gpt-5.4": _openai(2.50, 15.00),
        "gpt-5.4-mini": _openai(0.75, 4.50),
        "gpt-5.4-nano": _openai(0.20, 1.25),
        # Anthropic
        "claude-fable-5-1": _anthropic(10.00, 50.00),
        "claude-opus-5-5": _anthropic(4.00, 20.00),
        "claude-sonnet-5-5": _anthropic(2.00, 10.00),
        "claude-haiku-5-5": _anthropic(
            0.10, 0.50, long_context_threshold=100_000, long_input_per_mtok=0.50, long_output_per_mtok=2.50
        ),
        "claude-haiku-4-5": _anthropic(1.00, 5.00),
        # Groq (production models with public per-token prices; the Llama models are priced via sales)
        "openai/gpt-oss-120b": _groq_gpt_oss(0.15, 0.60),
        "openai/gpt-oss-20b": _groq_gpt_oss(0.075, 0.30),
    }
)


def get_pricing(model: str) -> Optional[ModelPricing]:
    return PRICING.get(model)


def require_pricing(model: str) -> ModelPricing:
    pricing = PRICING.get(model)
    if pricing is None:
        raise UnsupportedModelError(model, PRICING)
    return pricing


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return require_pricing(model).cost_usd(input_tokens, output_tokens)
