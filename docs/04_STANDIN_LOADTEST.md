# 04. Stand-in service, calibration, load scenarios, metrics, CSV

## 1. Stand-in service

A separate FastAPI app (`standin/service.py`, port 8081) that speaks **Groq's OpenAI-compatible wire format** (`POST /openai/v1/chat/completions`, SSE when `stream=true`, same response headers). Consequence: the *same* provider adapter (`providers/groq.py`) talks to both; only `base_url` and key differ. This removes a whole class of "works on the stand-in, breaks on Groq" bugs.

### 1.1 Required behaviours (brief B6) and how each is built

| Behaviour | Implementation |
|-----------|----------------|
| Slow | TTFT and decode time sampled from the calibrated distribution; `slow_success_prob` multiplies latency by `slow_factor` (success, just slowly) |
| Fail | `server_error_prob` (500/503), `timeout_prob` (accept then never answer until client gives up), `disconnect_prob` (drop mid-stream) |
| Refuse at a limit | Counters for RPM, TPM, RPD, TPD per model. Over limit -> HTTP 429 with `retry-after` and the same error JSON shape captured from the **real** 429 (copy it from `limit_probe.jsonl`; do not invent) |
| Slower under load | `latency *= 1 + alpha * max(0, (inflight - capacity) / capacity)` evaluated per call at start (and optionally re-evaluated per chunk) |
| Stream in pieces | SSE with N chunks; first chunk delayed by TTFT; final chunk carries `usage` the way Groq does (confirm shape with probe) |
| Mirror headers | `x-ratelimit-limit-requests` (RPD), `-remaining-requests` (RPD), `-reset-requests`, `x-ratelimit-limit-tokens` (TPM), `-remaining-tokens`, `-reset-tokens`, `retry-after` on 429 only, exactly as Groq documents them |
| Reproducible | Seeded RNG per scenario (`seed` in profile and manifest) |
| Controllable | `POST /admin/profile` (swap profile), `/admin/limits` (change limits mid-run, no notice to the client), `/admin/reset`, `GET /admin/state` |

### 1.2 What the stand-in returns (so the real pipeline runs unmodified)

It must work with the real orchestrator, which needs sane JSON:
- **Draft call** (detect by system prompt marker `RELAY_DRAFT_V1`): parse `<<chunk:ID>> ...` blocks from the user message; return a claim built from the first sentence of the top chunk, with `evidence=[{chunk_id, quote}]` where `quote` is a verbatim span of that chunk. So deterministic validation passes by construction when the stand-in is "well-behaved".
- **Verify call** (marker `RELAY_VERIFY_V1`): return `supported` for all claims, except with probability `verify_fail_prob` (default 0.05) mark one claim `unsupported` to exercise the repair loop.
- Completion token counts sampled from the real distribution per kind (draft/verify); input tokens counted by the same tokenizer as the app. `prompt_tokens_details.cached_tokens` simulated with a cache model: the static prefix is "cached" after first use for 2 h, mirroring Groq docs.

### 1.3 Profiles (`config/standin_profiles.yaml`)

| Profile | RPM | TPM | RPD | TPD | capacity (in-flight) | Used for |
|---------|-----|-----|-----|-----|----------------------|----------|
| `free_exact` | as on your Limits page | as on your Limits page | as on your Limits page | as on your Limits page | from real probe or "assumed 4" | fidelity check, S6 |
| `scaled` | exact x K | exact x K | exact x K | exact x K | 8 | S1-S5, S7 |

K = 30 by default. **Principle: scale every limit by the same factor so the binding limit is the same as in the real service (tokens).** The offered rates in Table 2a are therefore *stand-in rates*, not rates the free account can take; say so. Choose K so the sustainable rate S (below) lands near 80-100 answers/min, which makes 100-200 requests fit in 1.5-2.5 minutes per scenario.

## 2. Calibration

Inputs: `events.jsonl` rows from real calls (eval runs + probe + burst). Per call you need `ttft_ms`, `total_ms`, `in`, `out`, `cached`, `status`, headers, and Groq's own `usage.queue_time / prompt_time / completion_time` if present.

`standin/calibrate.py` produces `config/calibration.json`:

1. Per call kind (draft, verify): fit `ttft ~ LogNormal` from observed p50 and p95 (`sigma = (ln p95 - ln p50)/1.645`), and `decode tokens/sec ~ LogNormal` likewise. Prefer **bootstrap resampling of observed pairs** when n < ~50, with a lognormal tail beyond the max.
2. Completion-token distributions: empirical per kind.
3. Check: simulated p50/p95/p99 of total call time vs observed. Report the ratio. If p99 is off by more than 25%, say so; with ~100 real calls p99 is essentially the maximum, so write "n = ..., p99 unreliable".
4. Load slowdown `alpha`: from the real burst of 6 vs sequential calls: `alpha = (mean_latency_burst / mean_latency_seq - 1) / ((6 - capacity)/capacity)` if capacity < 6 and the ratio is > 1; otherwise record `alpha = assumed 0.5` and **label it assumed** in the write-up.
5. Failure probabilities: from real error counts if any; with ~100 calls you will likely have zero 5xx. Then the stand-in's fault rates are **chosen** (2-3% 5xx, 5-15% slow_success) and must be labelled "chosen, not measured".
6. Persist `seed`, `calibrated_from_run_ids`, `n_calls`, date.

**Do not call the stand-in "calibrated" for anything you did not fit from data.** Parts that are chosen are labelled chosen.

### 2.1 Where the stand-in differs most from Groq (write-up section 2 raw material)
- Latency under concurrency (Groq's internal batching) is a one-parameter guess from a 6-request burst.
- No cross-tenant noise, no cold starts, no regional/network variation, no model/routing changes.
- Fault model is independent per call; real faults correlate in time.
- Its limit counters are exact; Groq's real enforcement windows may be coarser/finer (we saw only headers).
- Token counts use our tokenizer, not Groq's.

## 3. Real-service runs (what the free account can take)

Budget at ~2.8K tokens/answer (see `07` for the day-by-day split):

| Run | Scenario name | n | How | Tokens (est.) |
|-----|---------------|---|-----|---------------|
| R0 | `probe` | ~45 tiny calls | `limit_probe.py`: RPM ramp until first 429, TPM probe, tiny concurrency probe | ~15K |
| R1 | `eval_real` | 18 x 3 settings = 54 | sequential, paced by our own limiter | ~140-160K |
| R2 | `real_burst6` | 6 | 6 simultaneous requests of the chosen setting | ~17K |

R1 is both your latency calibration data and your Table 3 data. Table 2a gets two real rows: `eval_real` (chosen setting only, n = 18) and `real_burst6`.

Never run a real daily-exhaustion test. Use the stand-in. Record "not observed (deliberately not exhausted)" in Table 1 for the TPD/RPD hit, and give the doc/console value with medium confidence.

## 4. Stand-in scenarios (all open-loop, seeded, cache OFF unless stated)

First run `loadgen capacity` on the `scaled` profile: ramp offered rate until success < 99% or p95 total > deadline. Call the highest good rate **S** (answers/min). All rates below are multiples of S and S goes in the write-up.

| ID | Name | Profile | Offered rate | Shape | Target n | What it should reveal |
|----|------|---------|--------------|-------|----------|----------------------|
| S1 | `steady` | scaled | 0.5 S | constant 150 s | ~100 | baseline; queue ~0 |
| S2 | `burst` | scaled | 0.5 S, then **8 x** that for 15 s, then 0.5 S | 45 s / 15 s / 45 s | ~140 | queue growth, early rejection, recovery time |
| S3 | `sustained_overload` | scaled | 1.75 S | 90 s | ~200 | bounded queue, shedding instead of backlog; p95/p99 stay bounded for accepted requests |
| S4 | `minute_limit_halved` | scaled | 0.8 S | 120 s; at t = 40 s `/admin/limits` halves **RPM and TPM** with no notice | ~160 | does header reconciliation / 429 handling adapt before wasting calls; who is delayed or dropped |
| S5 | `daily_exhausted` | scaled, TPD sized for ~60% of the run | 0.8 S | 100 s | ~135 | clean switch to zero-provider-call rejection; no retry of an impossible call; cache hits unaffected |
| S6 | `standin_burst6_exact` | free_exact | 6 simultaneous at t = 0 | one shot | 6 | **fidelity check** vs `real_burst6`: compare statuses, 429 count, latency order, retry-after |
| S7a | `retry_naive` | scaled + faults | 0.7 S | 90 s | ~100 | naive policy (call timeout 1.5 s, up to 3 retries, no jitter, no budget) under `slow_success_prob` 0.15 x8 and 3% 5xx |
| S7b | `retry_bounded` | scaled + same faults, same seed | 0.7 S | 90 s | ~100 | our policy; compare retries/request, p99, success % |
| S8 | `burst_cache_on` | as S2 with cache + single-flight ON | | | ~140 | tokens saved and rejections avoided |

Questions for the load tests are drawn from the 18 eval questions with a **Zipf-like** weight (a few questions dominate, like an incident) when cache is on; uniform when cache is off. Record the seed.

Multipliers (8x burst, 1.75x overload, halving) are initial choices. If S3 is not actually overloading (queue never fills, rejections ~0), raise the multiplier and record that you did. If it collapses instantly, lower it.

## 5. Metric definitions (put these in `aggregate.py`; they must match the write-up)

| Metric | Definition |
|--------|-----------|
| Requests sent | rows in scenario |
| Duration (s) | first scheduled arrival to last completion |
| Offered rate (/min) | requests sent / (span of scheduled arrivals in minutes). For one-shot S6, give "burst of 6" |
| First token p50/p95/p99 | numpy linear percentiles of `first_token_ms` over rows with `status=ok`; **state n** |
| Total time p50/p95/p99 | same over `total_ms` for `status=ok`; failures reported separately (count and their elapsed times) |
| Succeeded % | `ok` / sent |
| Rate-limited % | `rate_limited` / sent |
| Timed out or given up % | `timeout` / sent |
| (error %) | remainder; mention in a footnote so columns reconcile to 100 |
| Retries per request | sum(`retries`) / sent |
| Peak queue length | max `queue_len_at_arrival` (post-admission queue depth) |
| Tokens per answer | sum(tokens_in + tokens_out) over rows with `provider_calls > 0` and `status=ok` / count of those rows; cache hits excluded; refusals included; denominator stated |
| Cost per 1,000 answers | list-price equivalent: sum(uncached_in x P_in + cached x P_cached + out x P_out)/1e6 / n_ok x 1000. Real free-tier billed cost is $0. Report as "0.00 billed / X list-equiv". Prices from `config/prices.yaml` with the date read; stand-in rows use the real model's prices |
| Slow request | the request at or above p99 of `total_ms` among `ok` rows in a scenario that also shows a retry or a large `queue_wait_ms`; `timeline.py <rid>` prints its split |

Time units: ms with one decimal in the CSV (`float`). Percentages one decimal.

## 6. `results.csv` columns

Required, in this order (brief p.4):

`request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`

Extras appended (keep short; run-level config goes in `run_manifest.json`, joined by `run_id`):

`run_id, queue_wait_ms, retrieval_ms, provider_ttft_ms, provider_ms, retry_wait_ms, validation_ms, provider_calls, tokens_cached, failure_reason, http_status_last, queue_len_at_arrival, inflight_at_arrival, mode, cache_hit, model, citation_valid, retrieval_hit, deadline_ms`

Semantics:
- One row per **logical** request (including those we reject ourselves).
- `start_time`: ISO-8601 UTC with ms; the scheduled-and-sent arrival. Durations from monotonic clocks only.
- `first_token_ms`: arrival to first **user-visible** answer byte (after validation). Empty for rejected/failed rows. `provider_ttft_ms` is the internal Groq first-content-delta time.
- `total_ms`: arrival to final response or terminal failure (client-side in loadgen; in-process for eval).
- `retries`: additional attempts across the whole logical request (provider retries + repair counted separately in `provider_calls`).
- `tokens_in/out`: summed over all provider calls and attempts for the request, where usage is known. `tokens_in` includes cached tokens; `tokens_cached` separates them.
- `answer_grade`: filled for eval rows after grading; **empty for load-test rows** (they are not quality-graded). Declare this in the write-up.
- `retrieval_setting`: `A|B|C` (eval) or the setting used in load tests.

`run_manifest.json` per run: `run_id, scenario, source, profile (limits, capacity, alpha), seed, policy snapshot (policy.yaml), model, git commit, started_at, S`.

## 7. Raw artifacts to keep

`results/results.csv`, `results/events.jsonl` (all runs), `results/limit_probe.jsonl` (raw status/headers/body for each probe call, key redacted), `results/run_manifest.json`, `eval/grades.csv`, `results/tables.md` (generated). Table 1 is not computable from the CSV; it comes from `limit_probe.jsonl` and your Limits-page screenshot. Say so in the write-up.
