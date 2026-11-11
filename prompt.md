# prompt.md: seven build prompts for Codex

## How to use
- Put `AGENTS.md` at the repo root and the `docs/` folder alongside it. Codex should read them every run.
- **One prompt = one task = one commit.** Run in order. After each: run the acceptance commands yourself, read the diff, review with the checklist in `docs/07_PLAN_AND_RISKS.md` section 4, then commit with a message that records what you changed from Codex's output.
- Keep every Codex session export (real, uncleaned). You need them for the assistant-history section.
- If Codex's answer contradicts the docs, decide which is right with evidence, update the doc or the code, and note it in `NOTES.md`. That is your "overruled" or "caught" material.
- Why seven: P4 (pipeline) and P5 (control plane) are the two largest and most coupled pieces. Splitting them keeps each prompt under what one session can build and test well. Every prompt names its acceptance tests so the output is checkable.
- Prefix every prompt with: *"Read AGENTS.md and the docs listed under Read first. Ask no questions; state assumptions in your final message."*

---

## P1. Scaffold, ingestion, retrieval, offline retrieval eval

**Read first:** `AGENTS.md`, `docs/03_RETRIEVAL_EVAL.md`, `docs/02_ARCHITECTURE.md` (sections 4.1, 6), `docs/05_DEPLOYMENT_RUNBOOK.md` (sections 3, 4).

**Task**
1. Create the repo skeleton from `docs/02_ARCHITECTURE.md` section 6, `pyproject.toml` (deps: fastapi, uvicorn, httpx, pydantic>=2, pydantic-settings, pyyaml, pymupdf, rank-bm25, tiktoken, fastembed, numpy, pandas, pytest, pytest-asyncio, ruff), `Makefile` with the targets in the runbook (stub the ones for later prompts), `.env.example`, `.gitignore` (`.env`, `.venv`, `data/papers`, `data/indexes`, caches), `config/` YAML stubs.
2. `relay/retrieval/ingest.py` + `make ingest`: download the three PDFs at the exact URLs in `docs/03`, save to `data/papers/`, assert the arXiv version stamp on page 1 (fail loudly), write `data/corpus_manifest.json` (id, version, url, sha256, pages, fetched_at, `corpus_hash`).
3. `chunking.py`: reading-order text extraction with PyMuPDF blocks, section-path detection, paragraph-aware packing with overlap using tiktoken `o200k_base`, chunk kinds (`text|table|caption|references|appendix`), chunk ids `{paper}-v{ver}-p{page}-c{idx}`. Produce `data/chunks/{A,B,C}.jsonl` for settings A (256/40), B (512/64), C (384/48) and a D file identical to C.
4. `search.py`: `search_papers(query, paper_filter, top_k, setting) -> RetrievalResult` exactly as typed in `docs/02` 4.1 (explicit `empty` status). Setting A/B: BM25. C/D: BM25 + dense (fastembed `BAAI/bge-small-en-v1.5`) fused by reciprocal rank fusion; C adds a fastembed cross-encoder rerank of 12 candidates to 4 (if the cross-encoder cannot be loaded, fall back to RRF only and set `rerank_method="none"` in the result; never fail silently). `paper_hint` is a score boost (x1.3), not a filter. `numeric=true` adds the neighbouring chunk (same section) of the top hit.
5. `eval/retrieval_offline.py` + `make retr-eval`: reads `eval/gold.jsonl` (schema in `docs/03` section 5; create 2 clearly marked **FAKE** example rows so the code runs, and make the script refuse to run if fake rows are present in a non-test run) and prints hit@k, MRR, wrong-paper@1, and mean context tokens for settings A-D.
6. Real-call guard skeleton in `relay/providers/guard.py` per runbook section 3 (used by P2).

**Acceptance tests (pytest, no network except an opt-in marked `@pytest.mark.network` ingest test)**
- Version assertion fails when given a PDF with the wrong stamp (use a tiny fixture PDF).
- Chunk ids are unique, every chunk has paper/version/page/section fields, token counts are within target +- 15%.
- `search_papers` returns `status="empty"` (not an exception, not a missing key) for a nonsense query on a tiny fixture index.
- Setting C with no cross-encoder available reports `rerank_method="none"`.
- Guard refuses real calls unless `RELAY_ALLOW_REAL=1` and `--real`.

**Do not** hand-write gold answers or questions; do not add an LLM call anywhere.

---

## P2. Observability, ledger, Groq adapter, probes

**Read first:** `AGENTS.md`, `docs/02_ARCHITECTURE.md` (sections 3.2, 4.2-4.4, 5), `docs/01_DESIGN.md` (D8, D10), `docs/04_STANDIN_LOADTEST.md` (section 1, 3), `docs/05` (sections 2, 3).

**Task**
1. `relay/obs/clock.py` (injectable monotonic clock + wall clock for `start_time`), `relay/obs/events.py` (JSONL writer, schema in `docs/02` section 5, key redaction, 500-char body truncation).
2. `relay/control/ledger.py`: SQLite ledger with tables `calls(id, run_id, ts_utc, kind, model, tokens_in, tokens_out, tokens_cached, status, real)` and daily rollups by UTC date; functions `tokens_used_today(model)`, `requests_used_today(model)`, `record_call(...)`. Read-only helpers for the guard caps.
3. `relay/providers/base.py` + `groq.py`: `GroqCompatibleProvider(base_url, api_key, model)` used for **both** Groq and the stand-in. Async httpx client with **zero retries**, separate connect/read timeouts, `complete(LLMRequest) -> LLMResult` supporting non-stream and SSE stream (incremental deltas via callback; capture first reasoning delta time and first content delta time separately), cancellation (closing the stream on timeout), `response_format` json_schema with fallback to `json_object`, `reasoning_effort`, `max_completion_tokens`. Return: HTTP status, text, usage (prompt, completion, cached from `prompt_tokens_details.cached_tokens`, and Groq's `queue_time/prompt_time/completion_time` if present), a **subset of response headers** (`retry-after`, all `x-ratelimit-*`), timings (monotonic), `provider_ttft_ms`. Raise typed `ProviderError(kind=rate_limit_minute|rate_limit_day|server|timeout|auth|bad_request|disconnect, status, retry_after_ms, body_trunc)`. Real calls pass through `guard.py` and are recorded in the ledger.
4. `scripts/probe_groq.py` (one real call: capture raw non-stream response JSON and headers; one streamed call: capture the chunk shapes including where `usage` appears; test whether `json_schema` + `stream` work together and whether `reasoning_effort=low` is accepted; save to `results/probe_shapes.json`, key redacted).
5. `scripts/limit_probe.py --real` (confirm before sending; print estimated tokens/requests first): (a) RPM ramp with minimal prompts at a controlled rate until the first 429 or 40 requests; (b) TPM probe: two or three concurrent ~3K-token prompts to trigger a token 429, total <= 10K tokens; (c) concurrency probe: 8 simultaneous tiny calls, record per-call latency and status; (d) after any 429, poll with minimal calls to measure time until success and compare with `retry-after`. Write raw `status/headers/body_trunc/timings` per call to `results/limit_probe.jsonl`.
6. Provide `tests/fake_groq_server.py`: a minimal in-test HTTP server that can return 200 (stream and non-stream), 429 (minute and day variants), 500, and a hang, to test the adapter without network.

**Acceptance tests**
- Adapter returns `provider_ttft_ms` for a streamed fake response and sets it from the first *content* delta (reasoning deltas tracked separately).
- 429 with `retry-after: 2` becomes `ProviderError(kind=rate_limit_minute, retry_after_ms=2000)`; a body flagged as a daily limit becomes `rate_limit_day` (use a fixture; the real body shape is unknown until the probe runs, so make the classifier configurable and tell me what to fill in).
- No retries: a 500 from the fake server results in exactly one request received.
- Timeout cancels the stream and the fake server observes the disconnect.
- Authorization header never appears in any log or event file (grep test).
- Guard blocks real calls by default.

**Do not** use the Groq SDK; do not guess the 429 body shape; do not run `limit_probe.py` yourself (I will run it).

---

## P3. Stand-in service, profiles, calibration script

**Read first:** `AGENTS.md`, `docs/04_STANDIN_LOADTEST.md` (sections 1, 2), `docs/02` (section 4.2-4.4), the P2 adapter code and `results/probe_shapes.json` / `results/limit_probe.jsonl` **if they exist** (use real 429 shapes; otherwise make them configurable placeholders and say so).

**Task**
1. `standin/service.py` (FastAPI): `POST /openai/v1/chat/completions` in Groq's wire format incl. SSE with `usage` in the final chunk, mirrored `x-ratelimit-*` and `retry-after` headers (RPD/TPM semantics per docs), per-model RPM/TPM/RPD/TPD counters, load-dependent slowdown `1 + alpha * max(0, (inflight - capacity)/capacity)`, seeded RNG, fault injection (`server_error_prob`, `timeout_prob`, `disconnect_prob`, `slow_success_prob` x `slow_factor`), simulated prompt-prefix cache (`cached_tokens`), admin endpoints `/admin/profile`, `/admin/limits` (hot change, no notice to clients), `/admin/reset`, `/admin/state`.
2. Content generation per `docs/04` section 1.2: detect `RELAY_DRAFT_V1` / `RELAY_VERIFY_V1` markers in the system prompt, parse `<<chunk:ID>>` blocks, return schema-valid JSON whose evidence quotes are verbatim spans; verify returns `supported` except with `verify_fail_prob`. Output lengths sampled from distributions in the profile.
3. `standin/profile.py` + `config/standin_profiles.yaml`: `free_exact` (limits are inputs you fill from my Limits page; leave clearly marked placeholders) and `scaled` (every limit multiplied by `scale`, capacity 8). Every field is annotated `source: fitted | chosen | assumed`.
4. `standin/calibrate.py`: reads `results/events.jsonl` (`call_end` events with `source=real`), fits per-kind TTFT and decode-rate lognormals (or bootstrap if n < 50), empirical completion-token distributions, estimates `alpha` from a real burst if present else sets `alpha=0.5 assumed`, prints observed vs simulated p50/p95/p99 and the ratio, writes `config/calibration.json` with `n_calls` and a list of what was fitted vs chosen. It must print a warning when n < 100 for p99.

**Acceptance tests**
- Same seed -> identical latency sequence; different seed -> different.
- RPM/TPM/RPD/TPD each produce 429 with `retry-after` and correct mirrored headers; daily exhaustion returns the daily-type 429.
- `/admin/limits` halving takes effect on the next request and the `x-ratelimit-limit-tokens` header reflects it.
- With `capacity=2`, 6 concurrent requests are slower on average than 1; slowdown matches the formula within tolerance.
- Streaming: first chunk delayed by sampled TTFT; chunk count configurable; final chunk has usage.
- A provider adapter from P2 can complete a draft call against the stand-in with no code changes except `base_url`.
- `calibrate.py` on a synthetic `events.jsonl` recovers known lognormal parameters within 10%.

**Do not** hard-code latency constants; do not label anything "calibrated" unless it was fitted from data.

---

## P4. Pipeline core: interpret, tool, gate, draft, validate, repair, release

**Read first:** `AGENTS.md`, `docs/01_DESIGN.md` (D1-D6, D13), `docs/02_ARCHITECTURE.md` (sections 2, 4), `docs/03` (sections 3, 6).

**Task** (provider-agnostic; use the P2 provider and P1 retrieval; no admission control yet, that is P5)
1. `app/models.py`: Pydantic models for `Question`, `Interpretation`, `RetrievalHit/Result` (reuse P1), `DraftClaim`, `Draft`, `VerifyVerdict`, `Outcome` (final text, citations list, status, failure_reason, per-stage timings, token totals incl. all attempts, flags `retrieval_hit` not computed here).
2. `app/interpret.py`: local rules for `paper_hint`, `qtype`, `numeric`, `looks_out_of_scope`. No model call.
3. `tools/executor.py`: runs `search_papers` with timeout and optional fault injection (`TOOL_FAULT_PROB`, seeded), returns typed results; one local retry.
4. `app/gate.py`: `sufficient|weak|empty` from retrieval scores/coverage; broaden loop (max 2): add section-title terms, add adjacent chunks, drop paper boost; zero hits after broadening -> `REFUSED_LOCAL`.
5. Prompts in `app/prompts.py`: static system prompts with markers `RELAY_DRAFT_V1` and `RELAY_VERIFY_V1` and the JSON schemas; **static content first, variable content last** (cache-friendly); chunks rendered as `<<chunk:ID>> text <<end>>`; draft instructions: answer only from chunks, claims small and independent, quote verbatim <= 40 words and include qualifiers (theoretical vs measured, conditions), return `insufficient_evidence` when the chunks do not support an answer.
6. `app/validation.py`: deterministic checks (id was retrieved, quote is a normalised substring, numbers in claim appear in quotes, version present in metadata) then LLM verification on claims+quotes only; typed outcomes `pass | fail_repairable | fail_final | verify_invalid` (invalid is never a pass).
7. `app/render.py`: final text + citations built from chunk metadata, format `[Moshi v1, §4.2, p.7]`; refusal text for `insufficient_evidence` and for abstention ("could not verify"); no model-authored citation text.
8. `app/orchestrator.py`: state machine per `docs/02` section 2, at most one repair, deadline checks between stages, accumulates tokens over all attempts, emits events, `async handle(question, setting, request_id, deadline_ms) -> Outcome`. Streaming of the draft is internal; user-visible emission is only after validation (expose `first_user_token_at` and an async generator for the validated answer).

**Acceptance tests (use the P3 stand-in or an in-process fake provider)**
- Invariants from `docs/02` section 2: nothing emitted before validation; every citation's chunk id in retrieved set; quote substring check rejects a forged quote; version appears in every rendered citation.
- A forged `chunk_id` -> `fail_final` (or repairable once), never released.
- Verifier returning garbage -> `verify_invalid`, not pass; leads to repair/abstain.
- Out-of-library question with empty retrieval -> explicit refusal text, status ok, zero provider tokens.
- One repair max: stand-in with `verify_fail_prob=1` ends in abstention with exactly 1 repair and tokens attributed from all calls.
- Token accounting equals the sum of per-call usages across draft, verify, repair.
- Deadline exceeded mid-pipeline -> `timeout` outcome and no further provider calls.

**Do not** add retries, queues, or limiters here; do not let any code path return unvalidated model text.

---

## P5. Control plane and API: admission, limiter, retries, breaker, cache

**Read first:** `AGENTS.md`, `docs/01_DESIGN.md` (D7-D12), `docs/02_ARCHITECTURE.md` (section 3, invariants), `docs/04` (section 5 definitions of failure statuses).

**Task**
1. `control/limiter.py`: sliding 60 s token and request windows with reservations; `projected_wait(need)`, `reserve`, `reconcile(actual)`, `update_from_headers(...)` (overwrites remaining with header values; adopts changed `x-ratelimit-limit-tokens` immediately), cooling after 429 until `retry-after`.
2. `control/admission.py`: predictive admission per `docs/02` 3.1 with reasons `queue_full | projected_wait | daily_budget_exhausted | breaker_open`; chooses mode normal/pressure/exhausted from utilisation thresholds in `config/policy.yaml`; bounded queue; priority so `verify`/`repair` outrank new `draft`; semaphore for provider concurrency.
3. `control/retry.py`: orchestrator-owned retry policy: max 1 retry/request, global retry budget (token bucket, 10%), full-jitter backoff, `retry-after` honoured only if it fits `remaining_deadline - margin`, never retry daily-limit/auth/bad-request, never retry a slow call that is still running (cancel it, count its tokens). Also provide a `NaiveRetryPolicy` (timeout 1.5 s, 3 retries, no jitter, no budget) selectable by config for scenario S7a only.
4. `control/breaker.py` per design (open after 5 consecutive failures in 20 s, 5 s open, half-open single probe).
5. `app/cache.py`: key = sha256(normalised question | setting | model | prompt_version | corpus_hash); stores only validated outcomes; single-flight for concurrent identical keys; flag to disable (`--no-cache`, used by eval and S1-S7).
6. `app/api.py` (FastAPI, one worker): `POST /ask` (JSON in; SSE out; headers `X-Request-Id` accepted from client; on rejection HTTP 429/503 with JSON body {message, reason, retry_after_s if known}); `GET /healthz`; `GET /state` (queue length, inflight, limiter view, mode). Emits all events from `docs/02` section 5. Integrates the P4 orchestrator through the control plane. Degraded mode must reduce `top_k` and output caps per `docs/03` section 3 and skip repair.
7. User-facing messages in `app/messages.py` for each rejection reason, including the daily-exhausted text from `docs/06` section 7 (promise a reset time only if one is known).
8. All policy numbers read from `config/policy.yaml`; changing one value must change behaviour without code edits.

**Acceptance tests (stand-in, deterministic clock where possible)**
- Admission rejects (not queues) when `projected_wait > max_queue_wait`; no provider call and zero tokens for rejected requests.
- Header reconciliation: after a response with a lower `remaining-tokens`, the next admission uses the lower value; a mid-run change of `limit-tokens` is adopted without a 429.
- Daily exhaustion: after the ledger/headers say exhausted, **zero** provider calls are made; cache hits still return; message text matches.
- Retry policy: 429 with `retry-after` larger than remaining deadline -> no retry; 500 -> one jittered retry; daily 429 -> no retry; retry budget caps total retries across many failing requests.
- Slow-but-succeeding call under `NaiveRetryPolicy` creates duplicate in-flight calls; under the bounded policy it does not (this is the unit-level version of S7).
- Single-flight: 20 concurrent identical requests -> 1 provider pipeline execution.
- Breaker opens after 5 failures and recovers via half-open probe.
- Invariants: at most 1 retry, 1 repair; no emission before validation; no hidden retries (count requests at the fake server).

**Do not** add per-user fairness or extra queues; do not retry anywhere except `control/retry.py`.

---

## P6. Load generator, scenarios, aggregation, CSV, timeline

**Read first:** `AGENTS.md`, `docs/04_STANDIN_LOADTEST.md` (sections 4-7), `docs/00` (sections E, F), `docs/02` (section 5).

**Task**
1. `loadgen/arrivals.py`: open-loop schedules (constant, step/burst, ramp) generated from a seed; arrivals are scheduled by plan time and sent regardless of earlier completions; log `scheduled_arrival` vs `actual_send` to expose generator lag (warn if lag > 20 ms p95).
2. `loadgen/run.py`: CLI `loadgen run --scenario NAME`, `loadgen capacity` (ramp to find S), `loadgen real-burst --n 6 --real`. Async httpx client against the Relay API, consuming SSE; measures client-side `first_token_ms` (first byte of the answer event), `total_ms`, status mapping to {ok, rate_limited, timeout, error}; client patience = deadline (give up -> `timeout`, reason `client_gave_up`). Writes `results/client_rows.jsonl`. Starts/configures stand-in profile and admin hooks for scenarios (halve limits at t, set TPD) via `config/scenarios.yaml`.
3. `config/scenarios.yaml`: S1-S8 from `docs/04` section 4 as multiples of `S`; includes seeds, question sampling (uniform or Zipf), cache on/off, retry policy, fault profile, admin events (`at_s`, `action`).
4. `scripts/build_results_csv.py`: join `client_rows.jsonl` and eval-runner rows with server `events.jsonl` by `request_id`; write `results/results.csv` with the 12 required columns first in the exact order and the extras in `docs/04` section 6; validate allowed values; refuse duplicates.
5. `scripts/aggregate.py`: compute Tables 2a, 2b, 3 exactly per `docs/04` section 5 and `docs/03` section 6 definitions; print n next to every percentile; failures reported separately; write `results/tables.md` in the write-up's column order; also print Table 1 skeleton filled from `limit_probe.jsonl` where possible and `not measured` otherwise.
6. `scripts/timeline.py <rid>`: from `events.jsonl`, print the request's split into waiting (queue + limiter + semaphore), working (retrieval, provider calls, validation), retrying (backoff + repeated attempts), the first user-visible token time, and a plain sentence on whether the delay came before or after the first token.
7. `make scenario NAME=...` and `make scenarios` targets.

**Acceptance tests**
- Open-loop property: with a deliberately slow server, the number of requests *sent* by time t does not decrease (a closed-loop bug would fail this).
- Percentiles match `numpy.percentile(..., method="linear")` on a hand-made fixture; failures are excluded from latency percentiles and counted in outcome columns.
- CSV validator rejects bad `status`, bad `source`, missing required column, wrong column order, duplicate request ids.
- `timeline.py` on a fixture with one retry reports non-zero retrying time and the correct before/after-first-token statement.
- A full dry run of S1 at tiny scale completes against the stand-in and produces rows for every request, including rejected ones.

**Do not** compute table numbers anywhere except `aggregate.py`; do not use wall-clock for durations; do not average away tail latency.

---

## P7. Eval runner, grading tools, submission checks, adversarial review

**Read first:** `AGENTS.md`, `docs/03_RETRIEVAL_EVAL.md` (sections 5-6), `docs/05` (sections 4-7), `docs/07` (section 4).

**Task, part A: build**
1. `eval/runner.py`: `make eval-real SETTING=A|B|C [--real]`: runs the 18 questions **sequentially, in-process**, cache off, through the control plane (so our own limiter paces real calls), pre-checks the ledger and refuses to start if the run would exceed 70% of remaining TPD, writes `eval/runs/<run_id>.jsonl` with the answer text, rendered citations, retrieved chunk ids, per-stage timings, tokens, status, and `retrieval_hit` (all gold phrases present in retrieved chunks), then appends rows for `build_results_csv.py`.
2. `eval/grade.py`: terminal tool that shows question, gold claims, the system answer with citations and the cited quotes side by side, and records **my** grade (`correct|wrong|refused_correctly|refused_wrongly`) and failure stage (`retrieval_miss|model_misread|validator_false_reject|validator_miss|refusal_threshold|budget`) into `eval/grades.csv`. It must not suggest a grade. Include `grade.py --regrade 5` that picks 5 random rows for later re-grading and `grade.py --agreement`.
3. `scripts/check_submission.py` / `make check`: the checks listed in `docs/05` section 7 (CSV format; tables recomputed from CSV equal `results/tables.md`; secret scan for `gsk_`, Authorization, env values, and configurable personal strings across the submission folder and conversation exports; PDF page count <= 2 and all 12 headings in order if `writeup.pdf` exists).
4. `make pdf`: render `writeup.md` to a 2-page PDF with the CSS in `docs/06` (headless Chrome or WeasyPrint, whichever installs) and print the page count.

**Task, part B: review (separate Codex run, fresh context)**
"Act as a skeptical senior reliability engineer reviewing this repository against `docs/01_DESIGN.md`, `docs/02_ARCHITECTURE.md` and `docs/04_STANDIN_LOADTEST.md`. Look specifically for: unbounded queues; hidden retries; retry amplification; limiter races across `await`; misuse of Groq headers (`-requests` is per day, `-tokens` is per minute); first-token measured at the provider instead of after validation; tokens missing from failed/repair/verify calls; cached tokens not separated; closed-loop load generation; wall-clock used for durations; percentiles including failures; verifier pass on invalid output; citations sourced from model text; dropped arXiv versions; cache key gaps; tests that could call the real API; secrets in logs; a stand-in that is not seeded or whose parameters are labelled calibrated without being fitted. For each suspected defect write a **failing test first**, show it failing, then fix it, and list defects you could not confirm separately. Do not change behaviour that the docs specify."

**Acceptance tests**
- Runner refuses to start when the ledger shows insufficient TPD headroom (test with a fake ledger).
- `retrieval_hit` is computed from gold phrases and matches a hand-made fixture.
- `grade.py` never writes a grade without explicit input; agreement computation is correct on a fixture.
- Secret scan catches a planted fake key in a transcript fixture.
- Part B produces at least a written list of checked items and confirmed/unconfirmed defects, with failing tests committed for confirmed ones.

**Do not** auto-grade answers; do not edit gold files; do not fabricate agreement numbers.
