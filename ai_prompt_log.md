# AI Prompt Log

## Tool used

Claude Code (CLI), model Claude Opus 5.5 (`claude-opus-5-5`), working in this repo: it wrote the
code, ran `pytest`, inspected the installed SDKs and read the providers' pricing pages.
I reviewed the output with a separate Claude chat assistant and ran all live tests (`smoke.py`) myself.

## Prompt

Prompt 1, word for word (the two model placeholders were sent unfilled):

```
You are an expert Python AI Infrastructure Engineer. I need an asynchronous,
streaming multi-provider SDK wrapper supporting OpenAI (<CURRENT_OPENAI_MODELS>)
and Anthropic (<CURRENT_ANTHROPIC_MODELS>).

Requirements:
- Python 3.11+, typed with Pydantic v2.
- A class `UnifiedLLMGateway` with:
  async def stream(self, prompt: str, model: str, temperature: float = 0.7)
      -> AsyncGenerator[StreamChunk, None]
- Use the official async clients (AsyncOpenAI and AsyncAnthropic).

Normalization:
- Pydantic schema `StreamChunk` with: delta_text: str, provider: str,
  model: str, is_final: bool, telemetry: Optional[ChunkTelemetry].
- Hide all differences between OpenAI and Anthropic chunk structures
  from the consumer.

Token & cost telemetry:
- Static pricing dict with USD rates per 1,000,000 input and output tokens.
- OpenAI: stream_options={"include_usage": True}, and tiktoken for prompt
  token verification.
- Anthropic: capture input tokens from message_start, accumulate output
  tokens from message_delta events.
- On the final chunk (is_final=True), provide input tokens, output tokens,
  total latency (s), TTFT (ms), tokens/sec, and total cost in USD.

Error handling:
- Typed custom exceptions (GatewayProviderError, RateLimitExceededError,
  AuthenticationError). Handle rate limits, invalid keys, connection errors.
- Never leak raw API keys in logs, exceptions, or outputs.

Output clean code split into schemas.py, pricing.py, and gateway.py.
```

Later prompts:

2. After my first smoke run: use `Decimal` for cost, fix the `or` fallback, replace `KeyError` with a typed error, and note cache tokens in the README.
3. Add Groq as a third, OpenAI-compatible provider without changing the OpenAI or Anthropic paths.
4. Add `tests/test_pricing.py` and the README cache note, and fill in this log from what actually happened.
5. After my second smoke run (Anthropic + Groq): explain 77-vs-6 prompt tokens and 166 output tokens, quiet the mismatch warning, and document the reasoning-model definitions.
6. Summarize every file and check that none contains an API key, without opening `.env`.
7. Add the note that I reviewed with a separate Claude chat assistant.
8. Fix the log timeline, add an assignment checklist, and add the README test table.
9. Record the missing prompt, add the blocking-call finding, and check the malformed-key test. (I stopped this one partway.)
10. Rewrite this log to be short.
11. Fix the malformed-key gap with fake keys and a closed local port, add Test 3 checks, and fill in the README test table.
12. When building the app, create the gateway inside each request (`async with UnifiedLLMGateway() as gw`) and never cache it, to avoid "Event loop is closed" under Streamlit. (Interrupted before any change; applied in prompt 13.)
13. Build `src/app.py` in Streamlit: provider and model dropdowns, prompt, max tokens, temperature slider, token-by-token streaming, a telemetry panel with cost to six decimals, and short error messages with no keys or tracebacks; test the helpers in `tests/test_app.py`.

## What the AI produced

- **Code:** `src/schemas.py`, `src/pricing.py`, `src/exceptions.py`, `src/gateway.py`.
  - `UnifiedLLMGateway` streams OpenAI, Anthropic and Groq as one chunk format, and ends each stream with a telemetry chunk (tokens, latency, TTFT, tokens/sec, cost).
  - Errors are typed, and keys are redacted.
- **App:** `src/app.py` (Streamlit). A new gateway is created per run, never cached.
  - Completion tokens are labelled "includes reasoning tokens" / "estimated", and cost is shown to six decimals.
  - Typed errors become short messages. `.streamlit/config.toml` hides tracebacks as a backstop.
  - Not yet run against a live provider.
- **Tests:** 8 test files plus `tests/fakes.py`, using fake clients and fake keys (the key tests use real SDK clients pointed at a closed local port). The app's page tests run it headless, without reading `.env` or clicking Run.
  - `uv run pytest -v`: 98 passed, 1 skipped. The skipped test is Test 2, which runs only with `RUN_LIVE_TESTS=1` and `OPENAI_API_KEY`.
- **Pricing:** from the official pages: OpenAI and Anthropic verified 2026-10-18, Groq (console.groq.com/docs/models) 2026-10-19.
  - The prompt placeholders were filled from the installed SDKs and those pages.
- **`README.md`:** telemetry definitions, assignment test table, cost limits.

## Bugs found and fixed

| Bug                                                                                                                                                                                                                                                                                                                             | Why it matters                                                                                                                 | Fix                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cost printed `8.599999999999999e-06` (float arithmetic), seen in my smoke run                                                                                                                                                                                                                                                   | A money value with float noise, which builds up when costs are summed                                                          | `Decimal` maths rounded to 9 places; after the fix my smoke run printed `8.6e-06` (`test_cost_has_no_float_noise`)                                                                                                                                                                                                                                                                                                                    |
| `long_rate or base_rate` ignored a 0.0 long-context rate                                                                                                                                                                                                                                                                        | Charges the wrong rate with no error                                                                                           | `is not None` checks (`test_zero_long_tier_rate_is_honoured_not_replaced_by_base_rate`)                                                                                                                                                                                                                                                                                                                                               |
| `temperature` passed to Anthropic; SDK 1.15 rejects it with `TypeError`                                                                                                                                                                                                                                                         | Every Anthropic call would fail (per the SDK signature; not confirmed with a live rejected call)                               | `supports_temperature` flag; not sent to Anthropic                                                                                                                                                                                                                                                                                                                                                                                    |
| Spec said to add up `message_delta` output tokens, but each one is already a running total                                                                                                                                                                                                                                      | Double-counts output tokens and cost                                                                                           | Keep the latest value (`test_anthropic_stream_is_normalized_and_usage_is_cumulative`)                                                                                                                                                                                                                                                                                                                                                 |
| Unknown model raised a raw `KeyError`                                                                                                                                                                                                                                                                                           | Not a typed error; unhelpful on a typo                                                                                         | `UnsupportedModelError` listing supported models, raised before any API call (a `GatewayError`, not a `GatewayProviderError`)                                                                                                                                                                                                                                                                                                         |
| SDK exceptions carry key fragments and auth headers                                                                                                                                                                                                                                                                             | API key could leak in messages, logs or tracebacks                                                                             | Redact `sk-`/`gsk_`/Bearer/x-api-key, raise without chaining, store keys as `SecretStr` (`tests/test_security.py`)                                                                                                                                                                                                                                                                                                                    |
| tiktoken ≠ billed prompt tokens (Groq gpt-oss: 77 billed vs 6)                                                                                                                                                                                                                                                                  | Billing from tiktoken would under-charge; the warning fired on every Groq call; Test 2 can't pass on gpt-oss                   | Bill from provider usage; tiktoken only as fallback (`usage_estimated`) and diagnostic; `tiktoken_comparable` flag (debug log for gpt-oss and Claude)                                                                                                                                                                                                                                                                                 |
| Reasoning models: 166 output tokens for a 6-token reply, TTFT 643 of 664 ms                                                                                                                                                                                                                                                     | Looks like a counting bug                                                                                                      | Definitions documented in schema and README; `reasoning` flag; `reasoning_tokens` field                                                                                                                                                                                                                                                                                                                                               |
| Tokens/sec measured from the first token would overstate reasoning models                                                                                                                                                                                                                                                       | Hidden tokens come before any text                                                                                             | `tokens_per_second = output_tokens / total_latency_s`                                                                                                                                                                                                                                                                                                                                                                                 |
| Groq may report usage under `x_groq.usage`                                                                                                                                                                                                                                                                                      | Groq cost would always fall back to estimates                                                                                  | Read `usage`, then `x_groq.usage` (`test_groq_usage_under_x_groq_is_read`)                                                                                                                                                                                                                                                                                                                                                            |
| Fallback output-token count ran tiktoken on the event loop (found by the agent in the checklist pass)                                                                                                                                                                                                                           | tiktoken is CPU-bound and may download its encoding file on first use, which blocks every other coroutine                      | `asyncio.to_thread` (`gateway.py:282`); `test_gateway_makes_no_blocking_calls` would have failed on the old line                                                                                                                                                                                                                                                                                                                      |
| Malformed keys never raised `AuthenticationError` (found by the agent while checking Test 4). Offline check with a fake key and a closed local port, 2026-10-19: a newline or space gave `ProviderConnectionError`; a non-ASCII OpenAI key gave a raw `UnicodeEncodeError` whose `.object` held the whole `Bearer <key>` header | Assignment Test 4 couldn't pass, and a real key with a stray non-ASCII character would leak through the exception's attributes | Keys are stripped, then checked on first use for that provider (printable ASCII, no whitespace or control characters, not empty). Failures raise `AuthenticationError("<Provider> API key is empty or contains invalid characters")`, with no part of the key. As a safety net, any other non-SDK error while building or sending a request becomes a redacted `GatewayProviderError`, raised without chaining (`tests/test_keys.py`) |
| Prompts sent with unfilled model and price placeholders (my process issue)                                                                                                                                                                                                                                                      | Model IDs and prices depended on the AI's research                                                                             | Taken from the installed SDKs and official pricing pages, with sources and dates in `pricing.py`                                                                                                                                                                                                                                                                                                                                      |

## Not verified

- **Verified live:** Anthropic (`claude-haiku-5-5`) and Groq (`openai/gpt-oss-20b`), by me, 2026-10-19.
- **Not verified live:**
  - The OpenAI path: no key, so Test 2 hasn't run. Test 2 also differs from the assignment: it uses a long prompt instead of a 50-word one, is opt-in, and is skipped without an OpenAI key.
  - Whether Groq reports `reasoning_tokens`= 200.
  - The `temperature` rejection, which comes from the SDK signature only.
- **Inference only:** that gpt-oss's 77-token overhead is a hidden system prompt.
- **Not checked model by model:**
  - that every listed OpenAI model is a reasoning model;
  - whether some OpenAI reasoning models reject a non-default `temperature`.
- **Not priced:** cached-input, batch, priority, data-residency, OpenAI long-context.
