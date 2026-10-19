"""Tests for src/app.py helpers. `main()` is never called, so nothing loads .env or touches the network."""

from __future__ import annotations

import inspect

import pydantic
import pytest

from src import app
from src.exceptions import (
    AuthenticationError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    UnsupportedModelError,
)
from src.pricing import PRICING
from src.schemas import ChunkTelemetry, Provider, StreamChunk

FAKE_KEY = "sk-FAKEKEY1234567890"


def telemetry(**overrides) -> ChunkTelemetry:
    values = dict(
        input_tokens=77,
        output_tokens=166,
        total_latency_s=0.664,
        ttft_ms=643.0,
        tokens_per_second=250.0,
        cost_usd=5.5575e-05,
    )
    return ChunkTelemetry(**{**values, **overrides})


# --------------------------------------------------------------------------- dropdowns


@pytest.mark.parametrize("provider", list(Provider))
def test_models_are_filtered_by_provider(provider):
    models = app.models_for_provider(provider)
    assert models and models == sorted(models)
    assert all(PRICING[m].provider is provider for m in models)


def test_groq_models():
    assert app.models_for_provider(Provider.GROQ) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_temperature_is_ignored_exactly_for_models_that_do_not_support_it():
    ignored = {m for m, p in PRICING.items() if not p.supports_temperature}
    assert ignored == set(app.models_for_provider(Provider.ANTHROPIC))


# --------------------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (0.000142, "$0.000142"),
        (5.5575e-05, "$0.000056"),  # the live Groq run
        (8.6e-06, "$0.000009"),  # the live Anthropic run
        (0.0000005, "$0.000001"),  # half rounds up
        (0.0, "$0.000000"),
        (1.5, "$1.500000"),
    ],
)
def test_cost_has_exactly_six_decimals(cost, expected):
    assert app.format_cost(cost) == expected


@pytest.mark.parametrize(("ttft", "expected"), [(None, "n/a"), (643.4, "643 ms"), (1234.6, "1,235 ms")])
def test_format_ttft(ttft, expected):
    assert app.format_ttft(ttft) == expected


def test_format_tokens_per_second():
    assert app.format_tokens_per_second(250) == "250.0"
    assert app.format_tokens_per_second(1234.56) == "1,234.6"


@pytest.mark.parametrize(
    ("model", "estimated", "expected"),
    [
        ("openai/gpt-oss-20b", False, "Completion tokens (includes reasoning tokens)"),
        ("openai/gpt-oss-20b", True, "Completion tokens (includes reasoning tokens, estimated)"),
        ("claude-haiku-5-5", False, "Completion tokens"),
        ("claude-haiku-5-5", True, "Completion tokens (estimated)"),
    ],
)
def test_completion_tokens_label(model, estimated, expected):
    assert app.completion_tokens_label(PRICING[model], telemetry(usage_estimated=estimated)) == expected


def test_prompt_tokens_label():
    assert app.prompt_tokens_label(telemetry()) == "Prompt tokens"
    assert app.prompt_tokens_label(telemetry(usage_estimated=True)) == "Prompt tokens (estimated)"


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("exc", "expected_start"),
    [
        (AuthenticationError(f"Incorrect API key {FAKE_KEY}", provider=Provider.OPENAI), "OpenAI rejected the API key"),
        (RateLimitExceededError("slow", provider=Provider.GROQ, retry_after_s=2.0), "Groq rate limit reached. Try again in 2 s."),
        (ProviderConnectionError("boom", provider=Provider.ANTHROPIC), "Could not reach Anthropic"),
        (GatewayProviderError(f"bad {FAKE_KEY}", provider=Provider.OPENAI, status_code=500), "OpenAI returned an error (HTTP 500)."),
        (UnsupportedModelError("gpt-2", PRICING), "That model is not supported."),
        (RuntimeError(f"raw failure with {FAKE_KEY}"), "Something went wrong."),
    ],
)
def test_error_message_is_short_and_never_echoes_the_exception(exc, expected_start):
    message = app.error_message(exc)
    assert message.startswith(expected_start)
    assert "FAKE" not in message and "sk-" not in message
    assert "Traceback" not in message and type(exc).__name__ not in message


def test_input_validation_message_is_shown_but_pydantic_errors_are_not():
    assert app.error_message(ValueError("prompt must be a non-empty string")) == "prompt must be a non-empty string"

    with pytest.raises(pydantic.ValidationError) as info:
        StreamChunk(delta_text=FAKE_KEY, provider="nope", model="m")
    assert "FAKE" not in app.error_message(info.value)


# --------------------------------------------------------------------------- gateway lifetime


class _FakeOutput:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def markdown(self, text: str) -> None:
        self.calls.append(text)


class _FakeGateway:
    instances: list["_FakeGateway"] = []

    def __init__(self) -> None:
        self.entered = self.closed = False
        _FakeGateway.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True

    async def stream(self, prompt, model, temperature, *, max_tokens):
        for text in ("Hel", "lo"):
            yield StreamChunk(delta_text=text, provider=Provider.GROQ, model=model)
        yield StreamChunk(delta_text="", provider=Provider.GROQ, model=model, is_final=True, telemetry=telemetry())


@pytest.mark.asyncio
async def test_each_run_creates_and_closes_its_own_gateway(monkeypatch):
    monkeypatch.setattr(app, "UnifiedLLMGateway", _FakeGateway)
    _FakeGateway.instances = []

    output = _FakeOutput()
    first = await app._run("hi", "openai/gpt-oss-20b", 0.7, 64, output)
    await app._run("hi", "openai/gpt-oss-20b", 0.7, 64, _FakeOutput())

    assert len(_FakeGateway.instances) == 2
    assert all(g.entered and g.closed for g in _FakeGateway.instances)
    assert output.calls == ["Hel▌", "Hello▌", "Hello"]  # token by token, cursor removed at the end
    assert first.cost_usd == 5.5575e-05


def test_gateway_is_never_cached():
    source = inspect.getsource(app)
    assert "cache_resource" not in source and "cache_data" not in source and "session_state" not in source
    assert "async with UnifiedLLMGateway() as gw" in source


# --------------------------------------------------------------------------- page renders (headless)


@pytest.fixture
def page(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # never read .env in tests
    at = AppTest.from_file(str(app.Path(app.__file__)), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_page_renders_and_filters_models(page):
    page.selectbox[0].select(Provider.GROQ).run()
    assert page.selectbox[1].options == app.models_for_provider(Provider.GROQ)
    assert not page.slider[0].disabled


def test_temperature_slider_is_disabled_for_anthropic(page):
    page.selectbox[0].select(Provider.ANTHROPIC).run()
    assert page.slider[0].disabled
    assert any(c.value == app.TEMPERATURE_IGNORED for c in page.caption)


def test_run_button_is_disabled_without_a_prompt(page):
    assert page.button[0].disabled
