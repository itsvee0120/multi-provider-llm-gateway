# multi-provider-llm-gateway

An async streaming gateway (`UnifiedLLMGateway`) that puts OpenAI, Anthropic and Groq behind one
interface and attaches token, latency and cost telemetry to the final chunk of each stream.

## Run with Docker

1. Copy `.env.example` to `.env` and fill in the keys you have (`OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, `GROQ_API_KEY`). A provider without a key fails only when you pick it.
2. Run `docker compose up --build`.
3. Open http://localhost:8501.

Keys are passed to the container at runtime through `env_file: .env`. `.env` is listed in
`.dockerignore`, so it is never copied into the image.

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

## Assignment tests

Run with `uv run pytest -v`. Every test uses fake clients and makes no live API calls, except
Test 2. The key tests point real SDK clients at a closed local port (127.0.0.1:9) with fake keys.

| Assignment test | Checks | Test functions |
|---|---|---|
| **Test 1**: unified stream protocol | All providers yield the same `StreamChunk` shape: text deltas, then one final chunk with telemetry | `tests/test_gateway.py::test_openai_stream_is_normalized_with_final_telemetry`<br>`tests/test_gateway.py::test_anthropic_stream_is_normalized_and_usage_is_cumulative`<br>`tests/test_groq.py::test_groq_stream_is_normalized_and_skips_empty_chunks`<br>`tests/test_checklist.py::test_openai_skips_empty_choices_and_none_content` |
| **Test 2**: tiktoken within ±10% | tiktoken's prompt count is within ±10% of the billed prompt tokens | `tests/test_live_tokenizer.py::test_tiktoken_estimate_within_10_percent_of_billed_prompt_tokens` (**skipped without an OpenAI key**) |
| **Test 3**: telemetry sanity | On the final chunk, `cost_usd`, `tokens_per_second` and `ttft_ms` are all above zero, for OpenAI, Anthropic and Groq | `tests/test_checklist.py::test_final_chunk_telemetry_is_positive[openai\|anthropic\|groq]` |
| **Test 4**: missing or malformed keys | A missing or malformed key raises `AuthenticationError` | Missing: `tests/test_security.py::test_missing_openai_key_is_an_auth_error`, `::test_missing_anthropic_key_is_an_auth_error`, `tests/test_groq.py::test_missing_groq_key_fails_only_when_groq_is_requested`<br>Malformed (`""`, `"   "`, `"sk-FAKE KEY"`, `"sk-FAKÉKEY"` × 3 providers): `tests/test_keys.py::test_malformed_key_raises_authentication_error`<br>Rejected by the provider (401): `tests/test_security.py::test_invalid_openai_key_maps_to_auth_error_without_leaking`, `::test_invalid_groq_key_maps_to_auth_error_without_leaking` |
| **Test 5**: no key substrings in exceptions or logs | No part of a key appears in `str()`, `repr()`, attributes or captured logs, and nothing is chained (`__cause__`/`__context__` are `None`) | `tests/test_keys.py::test_malformed_key_raises_authentication_error`, `::test_trailing_newline_is_stripped_then_request_is_attempted`, `::test_unicode_error_while_building_request_is_typed_and_scrubbed`, `::test_unexpected_error_message_is_redacted`<br>`tests/test_security.py::test_invalid_openai_key_maps_to_auth_error_without_leaking`, `::test_invalid_groq_key_maps_to_auth_error_without_leaking`, `::test_rate_limit_carries_retry_after`, `::test_connection_error_mid_stream_is_typed`, `::test_gateway_repr_hides_keys`, `::test_redact_secrets_patterns` |

**Test 2 is skipped without an OpenAI key.** It's a live, opt-in test: it only runs with
`RUN_LIVE_TESTS=1` and `OPENAI_API_KEY` set. It uses a long (~1,100-token) prompt instead of the
assignment's 50-word one, so OpenAI's few fixed chat-format tokens stay inside ±10%. It runs only on
models flagged `tiktoken_comparable`, which excludes gpt-oss on Groq and Claude (see the tiktoken
row in `ai_prompt_log.md`).

**Test 3:** before this table was filled in, only `cost_usd > 0` was checked
(`tests/test_pricing.py::test_single_token_on_cheapest_model_is_not_rounded_to_zero`). Neither
`tokens_per_second > 0` nor `ttft_ms > 0` was asserted on a final chunk, so
`test_final_chunk_telemetry_is_positive` was added. It uses a fake clock so latency and TTFT are
never zero by chance.

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
