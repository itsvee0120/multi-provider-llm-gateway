# AI Prompt Log

## 1. Tool used

Claude Code (agentic CLI), model Claude Opus 5.5 (`claude-opus-5-5`), working directly in this
repository on Windows. It read and wrote files, ran `pytest` in the project's `.venv`, inspected
the installed SDKs, and fetched the providers' pricing and docs pages.

The work took three prompts. There is no separate "raw draft": the code was generated and tested
in the repo, and each prompt changed it in place. The process is described below as it happened.

## 2. Exact prompt

### Prompt 1: initial build

The placeholders `<CURRENT_OPENAI_MODELS>` and `<CURRENT_ANTHROPIC_MODELS>` were sent unfilled.

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

### Prompt 2: fixes after my review and a live smoke run

```
Things to fix

1. Float arithmetic (the audit finding from before).
(input_tokens * input_rate + output_tokens * output_rate) / 1_000_000 is what produced 8.599999999999999e-06. Compute with Decimal:

python
from decimal import Decimal

def _d(x: float) -> Decimal:
    return Decimal(str(x))

def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _d(self.input_per_mtok), _d(self.output_per_mtok)
    if (
        self.long_context_threshold is not None
        and input_tokens > self.long_context_threshold
    ):
        if self.long_input_per_mtok is not None:
            input_rate = _d(self.long_input_per_mtok)
        if self.long_output_per_mtok is not None:
            output_rate = _d(self.long_output_per_mtok)
    cost = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / Decimal(TOKENS_PER_UNIT)
    return float(round(cost, 9))

Round to 9 decimals here and to 6 only in the UI. If you round to 6 inside the gateway, a tiny request (1 token on a cheap model) becomes 0.0, and your Test 3 (total_cost_usd > 0) would fail.

2. The or fallback hides a bug.
self.long_input_per_mtok or input_rate falls back to the base rate if a long-tier rate is ever 0.0. The is not None checks above avoid that.

3. Unknown models raise KeyError.
The assignment wants typed domain exceptions, so a typo in a model name shouldn't give a raw KeyError. Have calculate_cost raise a GatewayProviderError subclass (e.g. UnsupportedModelError) with a message listing the supported models. Check what the gateway does when get_pricing returns None.

4. The docstring is honest about what's not modelled.
Batch, cached-input, and data-residency rates are skipped. That's fine for this assignment. One consequence is that if Anthropic usage ever reports cache tokens, your cost would be a bit off. You aren't using prompt caching, so just mention it in the README.

Add to the audit log
Finding: floating-point cost noise (8.599999999999999e-06) from the live run.
Fix: the Decimal computation above.
Evidence: the smoke-test output before and after, plus the pricing-page link and the date you verified.
```

### Prompt 3: add Groq

```
Add Groq as a third provider to my existing UnifiedLLMGateway without
changing the behaviour of the OpenAI and Anthropic paths.

Requirements:
- Groq exposes an OpenAI-compatible API. Reuse the AsyncOpenAI client with
  base_url="https://api.groq.com/openai/v1" and api_key from GROQ_API_KEY
  (stored as SecretStr). Create this client once in __init__, not per call.
- Add Provider.GROQ to schemas.py.
- Add pricing rows in pricing.py for <GROQ_MODEL_ID> at <INPUT_PRICE> input
  and <OUTPUT_PRICE> output USD per million tokens (verified <DATE>, source:
  Groq pricing page). Keep the Decimal cost calculation.
- Route by model name through the registry, same as the other providers.
- Streaming: use stream_options={"include_usage": True}. If Groq does not
  return usage on the final chunk, fall back to a tiktoken estimate and set
  a flag on telemetry saying the counts are estimated. Handle chunks where
  choices is empty or delta.content is None.
- Keep the same typed errors (AuthenticationError, RateLimitExceededError,
  GatewayProviderError) and the same key-redaction rules, including
  redacting strings that start with gsk_.
- Missing GROQ_API_KEY must raise a descriptive AuthenticationError only
  when a Groq model is requested, not when the gateway is created.
- Add fake-client tests for the Groq path, and a redaction test for gsk_.
```

Placeholders were again left unfilled; the agent filled them from Groq's docs (see finding 9).

### Prompt 4: review of my live smoke run (Anthropic + Groq)

```
Groq

It streamed, returned usage from the provider (usage_estimated=False), and the cost matches your table: 77 × $0.075 + 166 × $0.30 = $55.575 per million, which is 5.5575e-05. Two numbers need explaining, because a reader will ask about them.

1. Prompt tokens: 77 reported vs 6 by tiktoken.
Your prompt is about 6 tokens, so about 70 tokens are added by the provider. gpt-oss models use their own chat format, and it most likely injects a hidden system prompt. That's my inference, since I can't see inside Groq. The billed count is what you pay, so the gateway is right to treat it as the truth and tiktoken as a cross-check.

2. Output tokens: 166 for a 6-token reply.
These are almost certainly reasoning tokens. The model thought before it answered. That's why TTFT (643 ms) is nearly the whole request time (664 ms): the visible text only starts once the reasoning is done. Tokens/sec of 250 is billed output tokens divided by total time, so it includes hidden tokens. If the Groq response reports a reasoning-token count, you could show it separately. I haven't verified that it does.

What to do with these
Test 2 (±10% of tiktoken): it can't pass on Groq's gpt-oss models, because the gap is 70 tokens. That isn't a bug. Run that test on a model whose tokenizer matches tiktoken, and explain in the log why gpt-oss and Claude are excluded. Claude gets estimated_input_tokens=None anyway. You can't weaken the ±10% tolerance without saying so.
The warning fires on every Groq call. The 32-token mismatch warning will be noise here. Consider skipping it for models with known hidden overhead, or logging it at debug level. It also printed in the middle of your streamed text. That's cosmetic, but it will look messy in the UI, so make sure app.py doesn't show log lines.
UI wording: label completion tokens as "includes reasoning tokens" for reasoning models. Otherwise a grader will see 166 tokens for a short reply and think it's a bug.
Add to ai_prompt_log.md
Finding 1: fill in "After (smoke run)": 8.6e-06.
New finding: the tiktoken cross-check is invalid for models with hidden provider overhead (77 reported vs 6 estimated). The fix was to bill from provider usage and use tiktoken only as a fallback and diagnostic.
New finding: reasoning models make output tokens, TTFT, and tokens/sec mean something different. Document the definitions.
Mark Groq as verified live, on 2026-10-19.
```

## 3. Raw output summary

**After prompt 1.** Claude Code inspected the installed SDKs (openai 3.35.1, anthropic 1.15.0,
tiktoken 0.14.0, pydantic 2.14.1) for the current model IDs. It took prices from the OpenAI and
Anthropic pricing pages (verified 2026-10-18) and wrote:

- `src/schemas.py`: `Provider` enum, `StreamChunk`, `ChunkTelemetry` (adds an
  `estimated_input_tokens` field).
- `src/pricing.py`: a frozen `ModelPricing` dataclass and a read-only `PRICING` table covering
  11 OpenAI and 5 Anthropic models, with a long-context tier for `claude-haiku-5-5`. **Cost used
  plain float arithmetic.**
- `src/gateway.py`: the exceptions, key redaction, tiktoken counting and `UnifiedLLMGateway`.
  Streams yield text deltas, then one final chunk with empty text that carries the telemetry.
- `tests/fakes.py`, `tests/test_gateway.py`, `tests/test_security.py`: 18 tests with fake
  clients and no network calls.

During that same run, the first test run showed that the Anthropic SDK rejects `temperature` (finding 3).
That was fixed before the output was handed back. It also found that the SDKs use `httpx2`, not
`httpx`, and fixed the test fixtures.

**After prompt 2.** The exceptions moved into a new `src/exceptions.py`, because `pricing.py` now
raises `UnsupportedModelError` and importing it from `gateway.py` would be circular. `gateway.py` re-exports
them. `cost_usd` was rewritten with `Decimal`. I stopped the agent before it wrote
`tests/test_pricing.py` or ran `smoke.py`, because I want to run the smoke test myself.

**After prompt 3.** Groq was added: `Provider.GROQ`, two pricing rows, a Groq client built once in
`__init__`, and one OpenAI-compatible streaming path shared by OpenAI and Groq. Also a
`usage_estimated` telemetry flag, `gsk_` redaction, `tests/test_groq.py` and two Groq security
tests. 28 tests passed. Then `tests/test_pricing.py` and the README cost note were added:
34 tests passed.

**After prompt 4.** I ran `smoke.py` myself against Anthropic (`claude-haiku-5-5`) and Groq
(`openai/gpt-oss-20b`); the agent didn't make any live calls. The agent then:

- Added two flags per model in the pricing table: `reasoning` and `tiktoken_comparable`.
- Made the token-mismatch warning apply only to `tiktoken_comparable` models. For the others it
  is logged at debug level.
- Added a `reasoning_tokens` telemetry field and wrote out the telemetry definitions in the
  schema and the README.
- Wrote Test 2 as an opt-in live test, `tests/test_live_tokenizer.py`.

38 tests pass, and the live test is skipped by default.

## 4. Audit findings

Format: issue → why it matters → fix. "Found by" says whether the agent or I (the reviewer)
caught it.

### 1. Floating-point noise in cost (found by me, from a live smoke run)

- **Issue:** `cost_usd` computed `(input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`
  with floats. `smoke.py` (`claude-haiku-5-5`) reported a cost of `8.599999999999999e-06`.
- **Why it matters:** cost is a money value shown to users and summed across requests. Binary
  float error shows up in the output and builds up when costs are added together.
- **Fix:** compute in `Decimal`, converting rates via `Decimal(str(rate))` so `0.1` stays exactly
  `0.1`. Round to 9 decimal places in the gateway (`COST_DECIMAL_PLACES`), and round further only
  for display. Rounding to 6 places in the gateway would turn a 1-token request on a cheap model
  into `0.0`.
- **Evidence:**
  - Before: `8.599999999999999e-06` from my smoke run. The old formula gives exactly this value
    for 16 input + 14 output tokens at $0.10 / $0.50 per million:
    `(16*0.10 + 14*0.50)/1_000_000 → 8.599999999999999e-06`.
  - After (unit test): `calculate_cost("claude-haiku-5-5", 16, 14) == 8.6e-06` in
    `tests/test_pricing.py::test_cost_has_no_float_noise`, which passes.
  - After (smoke run, 2026-10-19): `8.6e-06`.
  - Prices: https://platform.claude.com/docs/en/about-claude/pricing (Claude Haiku 5.5:
    $0.10 input / $0.50 output per million tokens for prompts up to 100K), verified 2026-10-18.

### 2. `or` fallback ignored a zero long-context rate (found by me)

- **Issue:** `input_rate = self.long_input_per_mtok or input_rate`. A long-tier rate of `0.0` is
  falsy, so it silently fell back to the base rate.
- **Why it matters:** this charges the wrong rate with no error, and only on the long-context
  branch, which few requests reach.
- **Fix:** `if self.long_input_per_mtok is not None:` (and the same for output).
  `test_zero_long_tier_rate_is_honoured_not_replaced_by_base_rate` covers it.

### 3. The current Anthropic API has no `temperature` setting (found by the agent's tests)

- **Issue:** the first version passed `temperature` to `messages.create`. The first test run
  failed with `TypeError: AsyncMessages.create() got an unexpected keyword argument 'temperature'`.
  The installed SDK and the Messages API reference have no sampling settings.
- **Why it matters:** every real Anthropic call would have failed. The prompt's spec assumed an
  older API.
- **Fix:** a `supports_temperature` flag per model in the pricing table. `temperature` is sent and
  range-checked (0–2) only for models that accept it, and left out for Anthropic. A test asserts
  it is not sent.

### 4. Anthropic `message_delta` usage is cumulative (found by the agent)

- **Issue:** the prompt says to "accumulate output tokens from message_delta events", but
  `usage.output_tokens` on those events is already a running total.
- **Why it matters:** adding them up would double-count output tokens and overstate cost whenever
  more than one `message_delta` arrives.
- **Fix:** keep the latest value instead of adding. The test fixture sends two `message_delta`
  events (4, then 5) and asserts the total is 5.

### 5. Unknown model raised a raw `KeyError` (found by me)

- **Issue:** `calculate_cost` raised `KeyError` for a model not in the table.
- **Why it matters:** the assignment requires typed domain exceptions, and a typo should give an
  actionable message.
- **Fix:** `require_pricing()` raises `UnsupportedModelError`, with the bad name and the list of
  supported models. The gateway calls it before any provider request, so a typo never reaches the
  network.
- **Deviation:** I suggested making it a `GatewayProviderError` subclass. The agent made it a
  subclass of the base `GatewayError` instead: a bad model name isn't a provider failure, and code
  that catches `GatewayProviderError` to retry shouldn't retry a typo. I accepted that.

### 6. Raw SDK exceptions could carry credentials (found by the agent)

- **Issue:** the SDK exceptions hold the HTTP request and response, including auth headers.
  OpenAI's 401 message includes a partly hidden key (`sk-proj-****abcd`), and Groq's includes the
  key.
- **Why it matters:** the prompt requires that keys never leak. Chaining with `raise ... from exc`
  would still print the original exception in tracebacks and keep a reference to it.
- **Fix:**
  - Messages are scrubbed of the configured key values and of `sk-…`, `gsk_…`, `Bearer …` and
    `x-api-key` patterns.
  - The typed error is raised outside the `except` block, so `__context__` and `__cause__` are
    `None`.
  - Keys are held as `SecretStr`, and `__repr__` only says whether each provider is configured.
  - Tests check the message, repr, captured logs and exception chain for each provider.

### 7. tiktoken can't match the API's prompt count exactly (found by the agent)

- **Issue:** tiktoken counts the raw prompt, but the chat API also counts chat-formatting tokens.
  tiktoken also doesn't recognise `gpt-6-*` or `openai/gpt-oss-*`.
- **Why it matters:** treating the local count as the truth would under-bill.
- **Fix:**
  - The count the provider reports is used for cost.
  - The tiktoken count is kept in `estimated_input_tokens`, and a warning is logged if the API
    reports fewer tokens or more than 32 extra. Since prompt 4 this only applies to
    `tiktoken_comparable` models (see finding 11).
  - Unknown model names fall back to the `o200k_base` encoding.
  - The tiktoken count is only used for cost when the provider sends no usage, and then
    `usage_estimated=True` (added in prompt 3).

### 8. Tokens/sec definition (agent's design decision)

- **Issue:** measuring tokens/sec from the first token overstates throughput for reasoning
  models. They are billed for hidden tokens generated before any text appears.
- **Fix:** `tokens_per_second = output_tokens / total_latency_s`, documented on the field.

### 9. Unfilled placeholders in prompts 1 and 3 (process issue, mine)

- **Issue:** I sent `<CURRENT_OPENAI_MODELS>`, `<CURRENT_ANTHROPIC_MODELS>`, `<GROQ_MODEL_ID>`,
  `<INPUT_PRICE>`, `<OUTPUT_PRICE>` and `<DATE>` without filling them in.
- **Why it matters:** the model IDs and prices had to come from the agent's research, so they need
  checking.
- **Fix:** the agent took model IDs from the installed SDKs and prices from the official pages,
  and recorded the sources and dates in the `pricing.py` docstring:
  - OpenAI: https://developers.openai.com/api/docs/pricing (2026-10-18)
  - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (2026-10-18)
  - Groq: https://console.groq.com/docs/models (2026-10-19). The groq.com pricing page didn't load
    any prices. Only the two production models with public prices were added
    (`openai/gpt-oss-120b` $0.15 / $0.60, `openai/gpt-oss-20b` $0.075 / $0.30). The Llama models
    are "pricing via sales".
  - **Groq verified live on 2026-10-19:** in my `smoke.py` run, `openai/gpt-oss-20b` streamed,
    returned provider usage (`usage_estimated=False`), and cost `5.5575e-05`. That matches the
    table: 77 × $0.075 + 166 × $0.30 = $55.575 per million tokens.

### 10. Groq usage may come under `x_groq` (found by the agent)

- **Issue:** Groq has reported streaming usage under a non-standard `x_groq.usage` field instead
  of `usage`.
- **Why it matters:** if only `usage` were read, Groq costs would always fall back to estimates.
- **Fix:** `_usage_tokens()` checks `usage`, then `x_groq.usage`, then falls back to tiktoken with
  `usage_estimated=True`. `test_groq_usage_under_x_groq_is_read` covers it.

### 11. The tiktoken cross-check doesn't work for models with hidden provider overhead (found by me, from the live Groq run)

- **Issue:** on Groq, `openai/gpt-oss-20b` billed **77** prompt tokens for "Say hello in five
  words.", which tiktoken counts as **6**.
  - gpt-oss uses its own chat format, which most likely adds a hidden system prompt. That's an
    inference: Groq doesn't expose it.
  - The 32-token mismatch warning therefore fired on every Groq call, and it printed in the middle
    of the streamed text.
  - Test 2 of the assignment (tiktoken within ±10% of the billed count) can never pass on this
    model.
- **Why it matters:**
  - Billing from tiktoken would under-charge by about 92%.
  - A warning on every call is noise that hides real problems.
  - Running Test 2 on gpt-oss would report a bug that isn't there.
- **Fix:**
  - Billing always uses the provider's reported usage. tiktoken is only a fallback (with
    `usage_estimated=True`) and a diagnostic (`estimated_input_tokens`).
  - New `tiktoken_comparable` flag per model in `pricing.py`. It is `False` for gpt-oss on Groq
    (hidden overhead) and for Claude (different tokenizer). For Claude the gateway never computes
    a tiktoken count anyway, so `estimated_input_tokens` is `None`.
  - For those models the mismatch is logged at debug level. Models where tiktoken should match
    still warn.
  - Tests: `test_known_prompt_overhead_does_not_warn` (77 vs 6 on Groq: no warning, billed count
    used) and `test_unexpected_gap_still_warns_for_tiktoken_comparable_models`.
- **Test 2 scope:** it moved to `tests/test_live_tokenizer.py`, which only runs on
  `tiktoken_comparable` models (default `gpt-5.4-nano`, an OpenAI model).
  - gpt-oss and Claude are excluded for the reasons above.
  - The ±10% tolerance is **not** relaxed. Instead the prompt is long (1,107 tiktoken tokens), so
    OpenAI's few fixed chat-format tokens stay well under 10%. A short prompt would fail ±10% even
    on OpenAI.
  - The test is opt-in (`RUN_LIVE_TESTS=1` plus `OPENAI_API_KEY`) and **has not been run yet**: I
    don't have an OpenAI key.

### 12. Reasoning models change what output tokens, TTFT and tokens/sec mean (found by me, from the live Groq run)

- **Issue:** the same Groq run billed **166** output tokens for a 6-token reply. TTFT was
  **643 ms** of **664 ms** total, and tokens/sec was **250**.
  - The extra output tokens are almost certainly hidden reasoning, billed as output.
  - Visible text only starts once the reasoning finishes, so TTFT is close to total latency.
  - Tokens/sec counts the hidden tokens too.
- **Why it matters:** without definitions, a reader sees 166 tokens for a 5-word reply and assumes
  a counting bug.
- **Fix:**
  - **Definitions:** written out in `ChunkTelemetry` and in a "Telemetry definitions" table in the
    README:
    - `output_tokens` = billed tokens, including reasoning.
    - `ttft_ms` = time to the first *visible* token, including thinking.
    - `tokens_per_second` = billed output tokens ÷ total latency, so it isn't the speed of visible
      text.
  - **`reasoning` flag** per model, `True` for all listed OpenAI models and for gpt-oss. The UI
    should use it to label output tokens "includes reasoning tokens".
  - **`reasoning_tokens` field:** filled from `usage.completion_tokens_details.reasoning_tokens`
    when the provider sends it, otherwise `None`. Whether Groq sends it is **not verified**: the
    smoke output didn't show it, because the run happened before this field existed.
  - **Logs in the UI:** `src/app.py` is still empty, so there's no UI to check yet. When it's
    built, it must not show log output, and should label reasoning-model output tokens as above.

### Known limitations (not fixed)

- Cached-input, batch, priority, data-residency and OpenAI long-context rates are not modelled
  (see README). If Anthropic ever reported cache tokens, cost would be slightly off.
- Some OpenAI reasoning models may reject a non-default `temperature`. The gateway still sends it
  and turns any rejection into a `GatewayProviderError`.
- **Verified live:** Anthropic (`claude-haiku-5-5`) and Groq (`openai/gpt-oss-20b`), both on
  2026-10-19, through my `smoke.py` runs.
- **Not verified live:** the OpenAI path (no key yet), so Test 2 hasn't run either. The 38 unit
  tests all use fake clients.
- Marking every listed OpenAI model as a reasoning model is based on the GPT-5-and-later lineup,
  not checked model by model.
