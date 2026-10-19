from __future__ import annotations

import pytest

from src import gateway as gw
from src.gateway import UnifiedLLMGateway, UnsupportedModelError
from src.pricing import PRICING, calculate_cost
from src.schemas import Provider

from .fakes import FakeStream, anthropic_events, fake_anthropic, fake_openai, openai_chunk


async def collect(gen):
    return [chunk async for chunk in gen]


@pytest.fixture(autouse=True)
def fixed_tiktoken(monkeypatch):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 10)


@pytest.mark.asyncio
async def test_openai_stream_is_normalized_with_final_telemetry():
    stream = FakeStream([openai_chunk("Hel"), openai_chunk("lo"), openai_chunk(usage=(15, 2))])
    client = fake_openai(stream)
    gateway = UnifiedLLMGateway(openai_client=client)

    chunks = await collect(gateway.stream("hi", "gpt-5.4-mini", temperature=0.2))

    assert [c.delta_text for c in chunks] == ["Hel", "lo", ""]
    assert all(c.provider == Provider.OPENAI and c.model == "gpt-5.4-mini" for c in chunks)
    assert [c.is_final for c in chunks] == [False, False, True]
    assert all(c.telemetry is None for c in chunks[:-1])

    t = chunks[-1].telemetry
    assert (t.input_tokens, t.output_tokens, t.estimated_input_tokens) == (15, 2, 10)
    assert t.cost_usd == pytest.approx(calculate_cost("gpt-5.4-mini", 15, 2))
    assert t.ttft_ms is not None and t.total_latency_s >= t.ttft_ms / 1000

    call = client.chat.completions.create.calls[0]
    assert call["stream_options"] == {"include_usage": True}
    assert call["temperature"] == 0.2
    assert stream.closed


@pytest.mark.asyncio
async def test_openai_falls_back_to_tiktoken_when_usage_missing():
    gateway = UnifiedLLMGateway(openai_client=fake_openai(FakeStream([openai_chunk("x")])))
    final = (await collect(gateway.stream("hi", "gpt-5.5")))[-1]
    assert (final.telemetry.input_tokens, final.telemetry.output_tokens) == (10, 10)


@pytest.mark.asyncio
async def test_anthropic_stream_is_normalized_and_usage_is_cumulative():
    client = fake_anthropic(FakeStream(anthropic_events(["Hi", " there"], input_tokens=12, output_tokens=5)))
    gateway = UnifiedLLMGateway(anthropic_client=client)

    chunks = await collect(gateway.stream("hi", "claude-sonnet-5-5"))

    assert [c.delta_text for c in chunks] == ["Hi", " there", ""]
    assert all(c.provider == Provider.ANTHROPIC for c in chunks)
    t = chunks[-1].telemetry
    assert (t.input_tokens, t.output_tokens) == (12, 5)
    assert t.cost_usd == pytest.approx((12 * 2.00 + 5 * 10.00) / 1_000_000)
    call = client.messages.create.calls[0]
    assert call["max_tokens"] == gw.DEFAULT_ANTHROPIC_MAX_TOKENS
    assert "temperature" not in call  # rejected by the current Anthropic API


@pytest.mark.asyncio
async def test_breaking_early_closes_provider_stream():
    stream = FakeStream([openai_chunk("a"), openai_chunk("b"), openai_chunk(usage=(1, 2))])
    gateway = UnifiedLLMGateway(openai_client=fake_openai(stream))
    gen = gateway.stream("hi", "gpt-5.4")
    async for _ in gen:
        break
    await gen.aclose()
    assert stream.closed


@pytest.mark.asyncio
async def test_unsupported_model_is_rejected_before_any_call():
    with pytest.raises(UnsupportedModelError):
        await collect(UnifiedLLMGateway().stream("hi", "gpt-2"))


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [2.5, -0.1])
async def test_openai_temperature_is_validated(temperature):
    with pytest.raises(ValueError):
        await collect(UnifiedLLMGateway().stream("hi", "gpt-5.4", temperature=temperature))


def test_long_context_tier_applies_above_threshold():
    haiku = PRICING["claude-haiku-5-5"]
    assert haiku.cost_usd(100_000, 0) == pytest.approx(100_000 * 0.10 / 1_000_000)
    assert haiku.cost_usd(100_001, 1_000) == pytest.approx((100_001 * 0.50 + 1_000 * 2.50) / 1_000_000)
