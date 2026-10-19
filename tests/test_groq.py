from __future__ import annotations

import logging

import pytest

from src import gateway as gw
from src.exceptions import AuthenticationError
from src.gateway import GROQ_BASE_URL, UnifiedLLMGateway
from src.pricing import calculate_cost
from src.schemas import Provider

from .fakes import FakeStream, empty_delta_chunk, fake_openai, openai_chunk, x_groq_usage_chunk

GROQ_MODEL = "openai/gpt-oss-20b"


async def collect(gen):
    return [chunk async for chunk in gen]


@pytest.fixture(autouse=True)
def fixed_tiktoken(monkeypatch):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 10)


@pytest.mark.asyncio
async def test_groq_stream_is_normalized_and_skips_empty_chunks():
    stream = FakeStream(
        [
            empty_delta_chunk(),
            openai_chunk("Hel"),
            openai_chunk(text=None),  # choices == []
            openai_chunk("lo"),
            empty_delta_chunk(),
            openai_chunk(usage=(14, 3)),
        ]
    )
    groq = fake_openai(stream)
    openai_client = fake_openai(FakeStream([]))
    gateway = UnifiedLLMGateway(groq_client=groq, openai_client=openai_client)

    chunks = await collect(gateway.stream("hi", GROQ_MODEL, temperature=0.3))

    assert [c.delta_text for c in chunks] == ["Hel", "lo", ""]
    assert all(c.provider == Provider.GROQ and c.model == GROQ_MODEL for c in chunks)
    t = chunks[-1].telemetry
    assert (t.input_tokens, t.output_tokens, t.usage_estimated) == (14, 3, False)
    assert t.cost_usd == calculate_cost(GROQ_MODEL, 14, 3)

    call = groq.chat.completions.create.calls[0]
    assert call["stream_options"] == {"include_usage": True}
    assert call["temperature"] == 0.3
    assert openai_client.chat.completions.create.calls == []
    assert stream.closed


@pytest.mark.asyncio
async def test_groq_without_usage_falls_back_to_flagged_estimate():
    gateway = UnifiedLLMGateway(groq_client=fake_openai(FakeStream([openai_chunk("x"), empty_delta_chunk()])))
    t = (await collect(gateway.stream("hi", GROQ_MODEL)))[-1].telemetry
    assert (t.input_tokens, t.output_tokens, t.usage_estimated) == (10, 10, True)


@pytest.mark.asyncio
async def test_groq_usage_under_x_groq_is_read():
    gateway = UnifiedLLMGateway(groq_client=fake_openai(FakeStream([openai_chunk("x"), x_groq_usage_chunk(7, 2)])))
    t = (await collect(gateway.stream("hi", GROQ_MODEL)))[-1].telemetry
    assert (t.input_tokens, t.output_tokens, t.usage_estimated) == (7, 2, False)


@pytest.mark.asyncio
async def test_openai_models_never_touch_the_groq_client():
    groq = fake_openai(FakeStream([]))
    openai_client = fake_openai(FakeStream([openai_chunk("hi"), openai_chunk(usage=(1, 1))]))
    gateway = UnifiedLLMGateway(groq_client=groq, openai_client=openai_client)

    chunks = await collect(gateway.stream("hi", "gpt-5.4"))

    assert chunks[-1].provider == Provider.OPENAI
    assert chunks[-1].telemetry.usage_estimated is False
    assert groq.chat.completions.create.calls == []


def test_groq_client_is_built_once_at_init_with_groq_base_url():
    gateway = UnifiedLLMGateway(groq_api_key="gsk_FAKE_TESTKEY")
    client = gateway._groq_client
    assert str(client.base_url).rstrip("/") == GROQ_BASE_URL
    assert gateway._groq() is client


@pytest.mark.asyncio
async def test_missing_groq_key_fails_only_when_groq_is_requested(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    openai_client = fake_openai(FakeStream([openai_chunk("ok"), openai_chunk(usage=(1, 1))]))
    gateway = UnifiedLLMGateway(openai_client=openai_client)  # must not raise

    with pytest.raises(AuthenticationError, match="GROQ_API_KEY") as info:
        await collect(gateway.stream("hi", GROQ_MODEL))
    assert info.value.provider == Provider.GROQ

    assert (await collect(gateway.stream("hi", "gpt-5.4")))[0].delta_text == "ok"


@pytest.mark.asyncio
async def test_known_prompt_overhead_does_not_warn(monkeypatch, caplog):
    # Live Groq run: 77 prompt tokens billed for a prompt tiktoken counts as 6.
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 6)
    gateway = UnifiedLLMGateway(groq_client=fake_openai(FakeStream([openai_chunk("hi"), openai_chunk(usage=(77, 166))])))

    with caplog.at_level(logging.DEBUG, logger=gw.__name__):
        t = (await collect(gateway.stream("Say hello in five words.", GROQ_MODEL)))[-1].telemetry

    assert (t.input_tokens, t.estimated_input_tokens) == (77, 6)  # billed count wins
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("provider reported 77, tiktoken 6" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_unexpected_gap_still_warns_for_tiktoken_comparable_models(monkeypatch, caplog):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 6)
    gateway = UnifiedLLMGateway(openai_client=fake_openai(FakeStream([openai_chunk("hi"), openai_chunk(usage=(77, 1))])))

    with caplog.at_level(logging.WARNING, logger=gw.__name__):
        await collect(gateway.stream("hi", "gpt-5.4"))

    assert any("Prompt token mismatch" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(("reported", "expected"), [(150, 150), (None, None)])
async def test_reasoning_tokens_are_surfaced_when_reported(reported, expected):
    final = openai_chunk(usage=(77, 166), reasoning_tokens=reported)
    gateway = UnifiedLLMGateway(groq_client=fake_openai(FakeStream([openai_chunk("hi"), final])))
    t = (await collect(gateway.stream("hi", GROQ_MODEL)))[-1].telemetry
    assert t.reasoning_tokens == expected
    assert t.output_tokens == 166  # still the billed total
