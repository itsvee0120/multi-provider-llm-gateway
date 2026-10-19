# multi-provider-llm-gateway

An async streaming gateway (`UnifiedLLMGateway`) that puts OpenAI, Anthropic and Groq behind one
interface and attaches token, latency and cost telemetry to the final chunk of each stream.

## Telemetry definitions

The final chunk of every stream (`is_final=True`) carries a `ChunkTelemetry`:

| Field | Meaning |
|---|---|
| `input_tokens` | Billed prompt tokens **as reported by the provider**. Includes any template or hidden system prompt the provider adds. |
| `output_tokens` | Billed completion tokens. For reasoning models (`reasoning=True` in `pricing.py`: all listed OpenAI models and gpt-oss on Groq) this **includes hidden reasoning tokens**, so it can be far larger than the visible reply. |
| `reasoning_tokens` | The part of `output_tokens` spent on reasoning, if the provider reports it (`completion_tokens_details.reasoning_tokens`); otherwise `None`. |
| `ttft_ms` | Time from sending the request to the first **visible** text token. For reasoning models this includes all thinking time, so it can be close to `total_latency_s`. |
| `total_latency_s` | Time from sending the request to the end of the stream. |
| `tokens_per_second` | `output_tokens / total_latency_s`. End-to-end and including hidden reasoning tokens, so it is not the speed at which visible text appears. |
| `cost_usd` | Calculated from the billed `input_tokens` and `output_tokens`. |
| `estimated_input_tokens` | Local tiktoken count of the prompt text (OpenAI and Groq). A diagnostic only, never used for billing unless usage is missing. |
| `usage_estimated` | `True` when the provider sent no usage, so the token counts and cost are tiktoken estimates. |

Example from a live run (`openai/gpt-oss-20b` on Groq, 2026-10-19, prompt "Say hello in five words."):

- 77 input tokens were billed, but tiktoken counts 6.
- 166 output tokens were billed for a 6-token reply.
- TTFT was 643 ms out of 664 ms total.
- Tokens/sec was 250.
- Cost was $0.000055575 (`5.5575e-05`).

These numbers aren't a bug. The extra prompt tokens are probably a hidden system prompt in gpt-oss's chat format. That's an inference: Groq doesn't expose it. The extra output tokens are reasoning, and the reasoning is also why visible text only started just before the end.

When showing these numbers in a UI, label `output_tokens` "includes reasoning tokens" for reasoning models, and don't put log output in the UI.

## Cost calculation limits

Costs come from the static table in `src/pricing.py` (standard-tier USD per million tokens, with the
date each provider's prices were verified). Costs are calculated with `Decimal` and rounded to
9 decimal places. Round further only for display.

Not modelled: batch and priority tiers, cached-input rates, data-residency pricing, and OpenAI's
long-context (over 272K tokens) rates.

**Cache tokens:** the gateway doesn't use prompt caching, so Anthropic should never report cache
read or cache write tokens. If it ever does, those tokens aren't counted in `input_tokens`, so the
reported cost will be slightly off. Cache writes cost more than the base input rate and cache reads
cost less.
