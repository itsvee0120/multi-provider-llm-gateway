"""Provider-agnostic data models exposed to gateway consumers."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class ChunkTelemetry(BaseModel):
    """Usage, latency and cost for one completed stream. Attached to the final chunk only."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(
        ge=0, description="Billed prompt tokens as reported by the provider, including any hidden template."
    )
    output_tokens: int = Field(
        ge=0,
        description=(
            "Billed completion tokens. For reasoning models this includes hidden reasoning tokens, "
            "so it can far exceed the visible reply."
        ),
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Part of output_tokens spent on hidden reasoning, when the provider reports it; else None.",
    )
    total_latency_s: float = Field(ge=0, description="Request start to end of stream, in seconds.")
    ttft_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Time to the first visible text token in ms; None if no text was produced. For reasoning "
            "models this includes all thinking time, so it can be close to total latency."
        ),
    )
    tokens_per_second: float = Field(
        ge=0,
        description=(
            "Output tokens divided by total latency. End-to-end rather than measured from the first "
            "token, because reasoning models bill hidden tokens generated before any text appears."
        ),
    )
    cost_usd: float = Field(ge=0)
    estimated_input_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Local tiktoken count of the prompt text (OpenAI and Groq), for cross-checking reported usage.",
    )
    usage_estimated: bool = Field(
        default=False,
        description="True when the provider sent no usage, so token counts and cost are tiktoken estimates.",
    )


class StreamChunk(BaseModel):
    """One normalized unit of a streamed response, identical in shape for every provider."""

    model_config = ConfigDict(frozen=True)

    delta_text: str
    provider: Provider
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None
