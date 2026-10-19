"""Minimal stand-ins for the OpenAI and Anthropic async streaming clients."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any, Iterable, Optional

import httpx2 as httpx


class FakeStream:
    def __init__(self, events: Iterable[Any], fail_with: Optional[Exception] = None) -> None:
        self._events = list(events)
        self._fail_with = fail_with
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    def __aiter__(self) -> FakeStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            if self._fail_with is not None:
                raise self._fail_with
            raise StopAsyncIteration


class _Create:
    def __init__(self, result: FakeStream | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_openai(result: FakeStream | Exception) -> NS:
    return NS(chat=NS(completions=NS(create=_Create(result))), close=_noop)


def fake_anthropic(result: FakeStream | Exception) -> NS:
    return NS(messages=NS(create=_Create(result)), close=_noop)


async def _noop() -> None:
    return None


def openai_chunk(
    text: Optional[str] = None,
    usage: Optional[tuple[int, int]] = None,
    reasoning_tokens: Optional[int] = None,
) -> NS:
    choices = [] if text is None else [NS(index=0, delta=NS(content=text), finish_reason=None)]
    details = None if reasoning_tokens is None else NS(reasoning_tokens=reasoning_tokens)
    return NS(
        choices=choices,
        usage=None
        if usage is None
        else NS(prompt_tokens=usage[0], completion_tokens=usage[1], completion_tokens_details=details),
    )


def empty_delta_chunk() -> NS:
    """A chunk with a choice whose delta has no text, e.g. the role-only first chunk or a finish chunk."""
    return NS(choices=[NS(index=0, delta=NS(content=None), finish_reason="stop")], usage=None)


def x_groq_usage_chunk(prompt_tokens: int, completion_tokens: int) -> NS:
    """Final chunk shape where Groq reports usage under `x_groq` instead of `usage`."""
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return NS(choices=[], usage=None, x_groq={"usage": usage})


def anthropic_events(texts: list[str], input_tokens: int, output_tokens: int) -> list[NS]:
    events = [NS(type="message_start", message=NS(usage=NS(input_tokens=input_tokens, output_tokens=1)))]
    events.append(NS(type="content_block_start", index=0))
    events += [NS(type="content_block_delta", delta=NS(type="text_delta", text=t)) for t in texts]
    events.append(NS(type="content_block_stop", index=0))
    # Cumulative usage, sent twice to prove it is not summed.
    events.append(NS(type="message_delta", usage=NS(output_tokens=output_tokens - 1, input_tokens=None)))
    events.append(NS(type="message_delta", usage=NS(output_tokens=output_tokens, input_tokens=None)))
    events.append(NS(type="message_stop"))
    return events


def http_response(status: int, headers: Optional[dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://example.invalid"))


STEP_S = 0.1  # fake time that passes per streamed event


class FakeClock:
    """Stands in for `time` in the gateway module, so latency and TTFT are deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class ClockedStream(FakeStream):
    """Advances the fake clock by STEP_S before delivering each event."""

    def __init__(self, events: list[Any], clock: FakeClock) -> None:
        super().__init__(events)
        self._clock = clock

    async def __anext__(self) -> Any:
        self._clock.now += STEP_S
        return await super().__anext__()
