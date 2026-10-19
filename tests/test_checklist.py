"""Tests for the assignment checklist items that had no direct test (see ai_prompt_log.md, section 5)."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace as NS

import pytest

from src import gateway as gw
from src.gateway import UnifiedLLMGateway

from .fakes import (
    STEP_S,
    ClockedStream,
    FakeClock,
    FakeStream,
    anthropic_events,
    empty_delta_chunk,
    fake_anthropic,
    fake_openai,
    openai_chunk,
)



async def collect(gen):
    return [chunk async for chunk in gen]


@pytest.fixture(autouse=True)
def fixed_tiktoken(monkeypatch):
    monkeypatch.setattr(gw, "count_tokens", lambda text, model: 10)


# --------------------------------------------------------------------------- no blocking calls


def _calls(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                names.append(f"{func.value.id}.{func.attr}")
    return names


def test_gateway_makes_no_blocking_calls():
    tree = ast.parse(inspect.getsource(gw))
    calls = _calls(tree)

    assert "time.sleep" not in calls
    assert not [c for c in calls if c.startswith(("requests.", "urllib."))]
    # Only the async SDK clients are constructed.
    assert "OpenAI" not in calls and "Anthropic" not in calls
    # tiktoken encoding is CPU-bound (and may download its BPE file on first use), so it must only
    # run via asyncio.to_thread, never be called directly on the event loop.
    assert "count_tokens" not in calls
    assert calls.count("asyncio.to_thread") == 2


# --------------------------------------------------------------------------- clients created once


def test_openai_and_anthropic_clients_are_created_once_and_reused():
    gateway = UnifiedLLMGateway(openai_api_key="sk-TESTKEY-fake-openai", anthropic_api_key="sk-ant-TESTKEY-fake")
    assert gateway._openai() is gateway._openai()
    assert gateway._anthropic() is gateway._anthropic()


@pytest.mark.asyncio
async def test_injected_client_is_reused_across_streams():
    client = fake_openai(FakeStream([openai_chunk("a"), openai_chunk(usage=(1, 1))]))
    gateway = UnifiedLLMGateway(openai_client=client)
    await collect(gateway.stream("one", "gpt-5.4"))
    await collect(gateway.stream("two", "gpt-5.4"))
    assert len(client.chat.completions.create.calls) == 2
    assert gateway._openai() is client


# --------------------------------------------------------------------------- OpenAI empty chunks


@pytest.mark.asyncio
async def test_openai_skips_empty_choices_and_none_content():
    final_usage_chunk = openai_chunk(usage=(5, 2))  # choices == [], as sent with include_usage
    stream = FakeStream(
        [empty_delta_chunk(), openai_chunk("Hi"), openai_chunk(text=None), empty_delta_chunk(), final_usage_chunk]
    )
    gateway = UnifiedLLMGateway(openai_client=fake_openai(stream))

    chunks = await collect(gateway.stream("hi", "gpt-5.4"))

    assert [(c.delta_text, c.is_final) for c in chunks] == [("Hi", False), ("", True)]
    assert (chunks[-1].telemetry.input_tokens, chunks[-1].telemetry.output_tokens) == (5, 2)


# --------------------------------------------------------------------------- TTFT


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(gw, "time", NS(perf_counter=clock.perf_counter))
    return clock


@pytest.mark.asyncio
async def test_openai_ttft_starts_at_first_non_empty_text_delta(clock):
    events = [empty_delta_chunk(), openai_chunk(text=None), openai_chunk("Hi"), openai_chunk(usage=(1, 1))]
    gateway = UnifiedLLMGateway(openai_client=fake_openai(ClockedStream(events, clock)))

    t = (await collect(gateway.stream("hi", "gpt-5.4")))[-1].telemetry

    assert t.ttft_ms == pytest.approx(3 * STEP_S * 1000)  # the 3rd event is the first with text


@pytest.mark.asyncio
async def test_anthropic_ttft_starts_at_first_non_empty_text_delta(clock):
    # message_start, content_block_start, an empty text_delta, then "Hi" as the 4th event.
    events = anthropic_events(["", "Hi"], input_tokens=3, output_tokens=2)
    gateway = UnifiedLLMGateway(anthropic_client=fake_anthropic(ClockedStream(events, clock)))

    chunks = await collect(gateway.stream("hi", "claude-haiku-4-5"))

    assert [c.delta_text for c in chunks] == ["Hi", ""]
    assert chunks[-1].telemetry.ttft_ms == pytest.approx(4 * STEP_S * 1000)


# --------------------------------------------------------------------------- telemetry sanity (Test 3)


def _openai_events():
    return [openai_chunk("Hel"), openai_chunk("lo"), openai_chunk(usage=(12, 5))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_kwarg", "make_client", "events", "model"),
    [
        ("openai_client", fake_openai, _openai_events, "gpt-5.4-nano"),
        ("anthropic_client", fake_anthropic, lambda: anthropic_events(["Hel", "lo"], 12, 5), "claude-haiku-5-5"),
        ("groq_client", fake_openai, _openai_events, "openai/gpt-oss-20b"),
    ],
    ids=["openai", "anthropic", "groq"],
)
async def test_final_chunk_telemetry_is_positive(clock, client_kwarg, make_client, events, model):
    gateway = UnifiedLLMGateway(**{client_kwarg: make_client(ClockedStream(events(), clock))})

    final = (await collect(gateway.stream("hi", model)))[-1]

    assert final.is_final
    t = final.telemetry
    assert t.cost_usd > 0
    assert t.tokens_per_second > 0
    assert t.ttft_ms is not None and t.ttft_ms > 0
