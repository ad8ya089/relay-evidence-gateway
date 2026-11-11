# 01. Design

## 1. What we optimise (stated up front, because the brief says the goals conflict)

The company wants fast, good, cheap, and nobody turned away. It cannot have all four, and it never wants a wrong answer sent to stay fast. Our priority order:

1. **Never release an unverified or wrongly cited answer.** (Non-negotiable.)
2. **Accepted requests finish inside a deadline** (8 s). "Fast" is defined as this, not as a mean.
3. **Spend provider tokens only on work that can finish.**
4. **Serve as many requests as the above allow.** We accept turning some away during bursts.

What this costs: higher user-visible first-token time (we release after validation), some fast rejections under overload, and no streaming of unverified text. Each is measured and reported, not hidden.

## 2. The number that shapes everything: the real service is token-bound

From Groq's current rate-limit page (free tier, `openai/gpt-oss-20b`): 30 RPM, 1K RPD, 8K TPM, 200K TPD. **Confirm on your own Limits page**, because the page calls its table "a high level summary" with possible exceptions.

Per answer (draft call + verify call), target budget with the design below:

| Call | Input | Output | Notes |
|------|-------|--------|-------|
| Draft | ~1,600-2,000 (of which ~450 static prefix may be cached) | ~250 incl. reasoning at `low` | 4-6 chunks at 256-512 tokens |
| Verify | ~600 | ~80 | claims + quotes only |
| **Total** | **~2.2-2.6K in** | **~330 out** | **~2.6-3.0K per answer** |

Capacity of the *real* free account, with c = 2 calls/answer, T = ~2.8K tokens/answer:

- RPM-bound: 30 / 2 = 15 answers/min
- **TPM-bound: 8,000 / 2,800 = ~2.9 answers/min**
- RPD-bound: 1,000 / 2 = 500 answers/day
- **TPD-bound: 200,000 / 2,800 = ~71 answers/day**

So tokens bind in both windows, by a factor of five in the per-minute case and seven in the per-day case. This is the first "number we could not have guessed" candidate: *requests-per-minute, the limit everyone quotes, is not the one that binds.* Every design choice below follows from it: compact verifier, low reasoning effort, small `top_k`, prompt-prefix caching, answer cache, token reservations (not request counters) in the limiter, and degrading by shrinking context before anything else.

Caveat: cached prompt tokens do not count toward limits per Groq docs, so measured TPM/TPD use may be lower than the table above. Record `cached_tokens` per call and report the realised average.

## 3. Decisions and their defence

### D1. Application-led orchestration (not model-led)
The app owns the state machine, deadlines, budgets, retries and validation. The model does bounded tasks inside it (draft claims, judge support). *Why:* a model-led agent under rate limits retries and loops, burning scarce tokens, and you cannot explain a 12-second request. *Cost:* less flexible; fine for this task.

### D2. Release only validated answers (buffer, then stream)
We stream from Groq internally to measure `provider_ttft_ms`, but the person sees nothing until the answer passes validation; then we emit it as SSE. `first_token_ms` in `results.csv` is **user-visible** first token. *Why:* the brief forbids sending a wrong or embarrassing answer to stay fast. *Cost:* first token is ~one draft + one verify call later (we expect 2-4 s). We report both numbers so the cost of the decision is visible.

### D3. Evidence is quote-anchored; citations are rendered by code
Draft schema: each claim has `evidence: [{chunk_id, quote}]`. Deterministic checks: `chunk_id` was in the retrieved set; `quote` (normalised whitespace/hyphenation) is a substring of that chunk; numbers in the claim appear in the cited quotes. The user-facing citation `[Moshi v1, §4.2, p.7]` is built from chunk metadata, never from model text. *Why:* kills fabricated citations and wrong-version citations with zero tokens.

### D4. Verifier sees claims and quotes only
The LLM verifier gets `[{claim, quotes}]` and returns `supported | unsupported | overstated` per claim. *Why:* about a quarter of the draft's input cost. *Limit:* it cannot catch an error that is faithful to a quote but ignores a qualification elsewhere in the chunk (e.g. 160 ms theoretical vs ~200 ms practical). Mitigation: the draft prompt requires the quote to include the qualifier; the numeric check requires numbers to appear in quotes; the retrieval step pulls the adjacent chunk for numeric questions. Report residual errors honestly from manual grading.

### D5. One repair, no open loops
If validation fails with a *repairable* reason and the remaining deadline and budget allow, run one targeted repair (draft call given the validator's complaint, same evidence). Otherwise abstain with "could not verify". *Why:* each repair is another ~2K tokens; unbounded self-critique is how a token-bound service melts.

### D6. Evidence-sufficiency branch with a bounded broaden loop
After retrieval, a local gate decides `sufficient | weak | empty`. `weak/empty` triggers up to 2 local broadening steps (expand query with section-title terms, add adjacent chunks, drop the paper filter) before concluding. No LLM involved, so zero tokens. If still weak, we still let the draft call decide `insufficient_evidence` (cheap, ~1.5K tokens), because a local threshold alone would produce false refusals. *Exception:* retrieval returns literally zero hits, then we refuse locally.

### D7. Admission control is predictive and rejects at the door
On arrival compute `projected_wait` = time until the token bucket can satisfy this request's reservation (draft_est + verify_est + safety margin), given in-flight reservations. Admit only if `projected_wait <= max_queue_wait` **and** `projected_wait + expected_service <= remaining_deadline` **and** the queue is not full **and** the daily ledger can cover it **and** the breaker is not open. Otherwise reject immediately with `Retry-After`. *Why:* the brief says people give up fast; queueing a request that will miss its deadline wastes tokens *and* the person's time.

### D8. Reserve, then reconcile
Before admitting, reserve estimated tokens; on completion replace the reservation with actual `usage` (excluding cached tokens). After every response overwrite the local per-minute remaining with `x-ratelimit-remaining-tokens` and the per-day request count with `x-ratelimit-remaining-requests` (note: those two headers mean TPM and RPD respectively, per Groq docs). Per-day tokens have no header, so a SQLite ledger (UTC date key) is the only source; check it against the Groq console usage at the end of each day (this is our least certain assumption, see 8.2 below).

### D9. Three operating modes
- **Normal:** setting's full `top_k`, full output cap.
- **Pressure** (per-minute token utilisation >= 70% or daily consumption >= 80%): `top_k <= 3`, lower output cap, no repair, new admissions only. Validation is never skipped.
- **Exhausted** (daily budget gone, or breaker open): no provider call at all. Serve cache hits; reject everything else with a clear message and a reset time *only if* known from headers.

### D10. Retries: one owner, one budget
Only the orchestrator retries. The HTTP client has retries disabled. Policy: max 1 retry per request; global **retry budget** of 10% of attempts (token bucket); full-jitter backoff (base 200 ms, cap 1 s); honour `Retry-After` only if it fits inside `remaining_deadline - 1.5 s`; never retry daily-quota errors, auth/validation errors, or a *slow-but-still-running* call (we cancel by closing the stream, and count its tokens as spent). *Why:* Google SRE overload guidance and AWS retry guidance both warn about retry amplification. We demonstrate the failure with a naive policy in scenario S7.

### D11. Answer cache + single-flight
Normalised question + (setting, model, prompt_version, corpus_hash) is the key. Only validated outcomes are cached. Concurrent identical questions share one in-flight computation. *Why:* bursts are many people asking the same thing. *Honesty:* disabled in S1-S6 so numbers reflect worst-case; S7b shows the effect.

### D12. Open-loop load generation against the HTTP API
Arrival times are scheduled independently of completions (avoids coordinated omission). The load generator is a separate process calling the API over HTTP and measuring client-side first-token and total time. Client patience equals the deadline; a request not finished by then is `timeout` (given up).

### D13. Local-only question interpretation
Regex/keyword rules decide: paper hints (Transformer/Attention, Moshi, dynamic chunking / H-Net), question type (explain / compare / number / limitation), and whether the question looks out of scope. No model call. *Why:* a serial planner call costs ~1K tokens of an 8K/min budget.

## 4. Parameters (all in `config/policy.yaml`; every one is an engineering choice, not a fact from a source)

| Parameter | Start value | Revisit after |
|-----------|-------------|---------------|
| `deadline_ms` | 8000 | real latency p95; if real p95 total > 6 s, raise |
| `max_queue_wait_ms` | 3000 | S2 burst results |
| `queue_capacity` | 20 (stand-in), 6 (real) | peak queue in S3 |
| `max_inflight_provider` | 4 stand-in (profile sets it), 2 real | real burst of 6 |
| `draft_max_completion_tokens` | 450 normal / 280 pressure | observed `completion_tokens` p95 |
| `verify_max_completion_tokens` | 120 | observed |
| `reasoning_effort` | low | answer quality; raise only if wrong-answer rate is high |
| `call_timeout_ms` | draft 5000, verify 3000 | 2 x real p99 (small n, state the caveat) |
| `retry.max_per_request` | 1 | S7 |
| `retry.budget_fraction` | 0.10 | S7 |
| `repair.max` | 1 | quality results |
| `pressure.minute_util` / `pressure.day_util` | 0.70 / 0.80 | S3, S5 |
| `breaker` | open after 5 consecutive failures in 20 s; half-open probe after 5 s | S4 |

Any number you change after seeing data becomes part of your write-up story ("measurement X contradicted default Y, so I changed it"). The live interview asks you to change the system; keep these one-line edits.

## 5. Failure taxonomy (what the CSV `status` and `failure_reason` mean)

| status | failure_reason examples | Meaning |
|--------|------------------------|---------|
| ok | (empty) | validated answer or validated refusal delivered |
| rate_limited | `queue_full`, `projected_wait`, `daily_budget_exhausted`, `provider_429_minute`, `provider_429_unrecoverable` | rejected by our admission control or by Groq's limit |
| timeout | `deadline`, `call_timeout`, `client_gave_up` | not finished by the deadline |
| error | `provider_5xx`, `validation_failed_after_repair`, `schema_invalid`, `tool_error`, `breaker_open` | everything else |

Note that `validation_failed_after_repair` returns the explicit "could not verify" message to the person; it is still `error` for accounting (we did not deliver an answer), but is **not** counted as `wrong`.

## 6. What we deliberately do not build (candidates for write-up section 8)

- A UI or chat front end. Not evaluated.
- Distributed queue/broker (Redis, Kafka), hosted vector DB, LangChain/LangGraph, tracing SaaS. Obscure where a delay or retry came from.
- Priority classes / per-user fairness (SRE "criticality"). Natural next step; out of scope.
- Docker/Kubernetes deploy. Local processes are enough and simpler to reproduce.

Pick **two** for the write-up and say what each would cost you.

## 7. Defending the stack

Python 3.11+, FastAPI, asyncio, httpx, Pydantic v2, SQLite, PyMuPDF, rank-bm25, tiktoken, fastembed (ONNX embeddings + cross-encoder rerank), numpy/pandas, pytest. *Why Python:* the bottleneck is the network and the provider, not CPU; C++/Go give nothing measurable here. *Why raw httpx, not the Groq SDK:* we need response headers, per-call timings, streaming deltas and no hidden retries. *Why SQLite:* durable daily ledger and run metadata with no service to run.

## 8. Honest unknowns (write-up section 8 raw material)

1. **Stand-in fidelity.** It cannot reproduce Groq's internal batching, cold starts or shared-capacity effects. Largest expected gap: how latency degrades with concurrency (we calibrate it from one real burst of 6, which is tiny).
2. **TPD accounting.** We assume the daily limit counts `usage.total_tokens` minus cached tokens on successful calls only. If 429s or reasoning tokens are counted differently the ledger drifts and the system either sheds too early or hits a real 429. Check the Groq console at day end.
3. **p99 on small n.** The real runs have tens of calls; p99 is nearly the max. State n next to every percentile.
4. **Grading subjectivity.** One human grader (you). Re-grade a random 5 two hours later and report agreement.
