from __future__ import annotations

import pytest

from src.exceptions import GatewayError, UnsupportedModelError
from src.pricing import PRICING, ModelPricing, calculate_cost
from src.schemas import Provider


def test_cost_has_no_float_noise():
    # 16 in + 14 out on claude-haiku-5-5 came out as 8.599999999999999e-06 with float maths.
    assert calculate_cost("claude-haiku-5-5", 16, 14) == 8.6e-06


def test_fractional_rates_are_exact():
    # 0.075 has no exact binary float representation.
    assert calculate_cost("openai/gpt-oss-20b", 3, 7) == 2.325e-06


def test_single_token_on_cheapest_model_is_not_rounded_to_zero():
    cheapest = min(PRICING, key=lambda m: PRICING[m].input_per_mtok)
    assert calculate_cost(cheapest, 1, 0) > 0


def test_long_context_tier_applies_only_above_threshold():
    assert calculate_cost("claude-haiku-5-5", 100_000, 1_000) == 0.0105  # base 0.10 / 0.50
    assert calculate_cost("claude-haiku-5-5", 100_001, 1_000) == 0.0525005  # long 0.50 / 2.50


def test_zero_long_tier_rate_is_honoured_not_replaced_by_base_rate():
    pricing = ModelPricing(
        Provider.OPENAI,
        1.00,
        2.00,
        long_context_threshold=10,
        long_input_per_mtok=0.0,
        long_output_per_mtok=0.0,
    )
    assert pricing.cost_usd(10, 5) == 0.00002  # at the threshold: base rates
    assert pricing.cost_usd(11, 5) == 0.0  # above it: the 0.0 tier applies


def test_unknown_model_raises_typed_error_listing_supported_models():
    with pytest.raises(UnsupportedModelError) as info:
        calculate_cost("gpt-5.4-mni", 1, 1)
    assert isinstance(info.value, GatewayError)
    assert info.value.model == "gpt-5.4-mni"
    assert "gpt-5.4-mini" in str(info.value)
