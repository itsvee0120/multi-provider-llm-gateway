"""Live check that tiktoken's prompt count is within 10% of the provider's billed count.

Opt-in only, because it makes a real (cheap) API call:
    RUN_LIVE_TESTS=1 OPENAI_API_KEY=... pytest tests/test_live_tokenizer.py

Only models flagged `tiktoken_comparable` are valid targets:
  - Claude uses its own tokenizer; the gateway never computes a tiktoken estimate for it
    (estimated_input_tokens is None).
  - gpt-oss on Groq bills a large hidden prompt overhead (77 tokens for a ~6-token prompt in a
    live run on 2026-10-19), so the gap is not a tokenizer error.

The prompt is deliberately long (~1,000 tokens): chat formatting adds a few fixed tokens
even on OpenAI, which would exceed 10% on a short prompt. The 10% tolerance is not relaxed.
"""

from __future__ import annotations

import os

import pytest

from src.gateway import UnifiedLLMGateway
from src.pricing import PRICING

TOLERANCE = 0.10
MODEL = os.environ.get("LIVE_TOKENIZER_MODEL", "gpt-5.4-nano")
PROMPT = "Summarise in one word. " + " ".join(f"Item {i} is a plain sentence about apples." for i in range(110))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="live test: set RUN_LIVE_TESTS=1 and OPENAI_API_KEY",
)


@pytest.mark.asyncio
async def test_tiktoken_estimate_within_10_percent_of_billed_prompt_tokens():
    assert PRICING[MODEL].tiktoken_comparable, f"{MODEL} is not a valid target for this check"

    async with UnifiedLLMGateway() as gateway:
        chunks = [c async for c in gateway.stream(PROMPT, MODEL, max_tokens=256)]

    t = chunks[-1].telemetry
    assert not t.usage_estimated, "provider sent no usage, so there is nothing to compare against"
    assert t.estimated_input_tokens is not None
    gap = abs(t.input_tokens - t.estimated_input_tokens) / t.input_tokens
    assert gap <= TOLERANCE, f"billed {t.input_tokens}, tiktoken {t.estimated_input_tokens} ({gap:.1%})"
