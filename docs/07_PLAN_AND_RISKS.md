# 07. Plan, token budget, risks

## 1. Calendar (all times IST)

- Today: **Wed 11 Nov**. Deadline: **Sat 14 Nov, 09:38**. Target submit: **Fri 13 Nov, 23:00**. Saturday morning = buffer for fixes only.
- Groq daily limits reset at **00:00 UTC = 05:30 IST** per a secondary source (confirm with `x-ratelimit-reset-requests` on a fresh day). The UTC "days" you can use:

| UTC day | IST window | Real-token plan (TPD 200K assumed; check yours) |
|---------|-----------|--------------------------------------------------|
| Nov 11 | now -> Thu 05:30 | Probe (~15K) + <= 10 real smoke answers (~28K). Total <= ~45K. Do not run evals before the pipeline is validated on the stand-in. |
| Nov 12 | Thu 05:30 -> Fri 05:30 | `eval-real` A and B (36 runs, ~100K) + <= 40K reruns |
| Nov 13 | Fri 05:30 -> Sat 05:30 | `eval-real` C (18, ~50K) + real burst of 6 (~17K) + <= 60K reruns/regrades |
| Nov 14 | Sat 05:30 -> 09:38 | Emergency only. Do not plan on it. |

Rules: stop a real run if the ledger shows < 25% of TPD remaining; never exhaust the real daily budget on purpose (use the stand-in for that).

Because Table 3 needs C before you can choose a setting, the "chosen setting" real burst is run on Fri after C. If C is not ready by Fri 10:00, run the burst with the better of A/B and say so.

## 2. Work blocks (about 8 hours)

| # | Block | Time | Prompt | Output / exit test |
|---|-------|------|--------|--------------------|
| 1 | Groq account, Limits screenshot, repo scaffold, `.env`, real-call guard | 30 min | P1 (scaffold part) | guard test passes; screenshot saved |
| 2 | Ingest 3 PDFs, chunking, indexes, offline retrieval eval harness | 60 min | P1 | version assertion passes; `make retr-eval` prints table |
| 3 | **You write 18 questions + gold keys from the PDFs** | 45 min | none | `questions.jsonl`, `gold.jsonl`, spot-checked against PDFs |
| 4 | Provider adapter, events, ledger, probe scripts; run probe | 50 min | P2 | `limit_probe.jsonl`, Table 1 draft |
| 5 | Stand-in service + profiles (uncalibrated first) | 50 min | P3 | tests for slow/fail/429/stream/load pass |
| 6 | Pipeline core (interpret, tool, gate, draft, validate, repair, release) | 60 min | P4 | invariants tests pass on stand-in; 3 smoke answers on real |
| 7 | Control plane (admission, limiter, retries, cache, API) | 60 min | P5 | admission and retry tests pass |
| 8 | Real eval A, B (background while you do 9) | wall-clock ~35 min | P7 runner | `eval/grades` for A, B |
| 9 | Loadgen, scenarios, aggregation, CSV, timeline | 60 min | P6 | `make aggregate` prints Tables 2a/2b |
| 10 | Calibrate stand-in; capacity S; run S1-S8 | 40 min | - | manifests + events for all scenarios |
| 11 | Real eval C, real burst 6; grade (your own); regrade 5 | 60 min | - | Table 3 complete |
| 12 | Final adversarial review pass; fix confirmed defects | 30 min | P7 review part | failing tests -> fixed |
| 13 | Write-up (2 pages), `make check`, conversation exports, Drive upload | 60-75 min | - | confirmation email |

Total about 8 hours of attention, some blocks running in the background. If you fall behind, cut in this order: S8 cache scenario, offline D config, the real burst (use S6 only and say so), the 4th retrieval metric. **Do not cut**: the five required scenarios, three LLM-graded settings, 18 questions, raw files, the write-up headings.

## 3. What AI builds vs what you own

| AI builds | You decide, write, or verify |
|-----------|-----------------------------|
| PDF parsing, chunking, BM25/hybrid code | Version pins, extraction quality, page numbers, chunk-size choices and why |
| Provider adapter, SSE parser, stand-in service | Model choice, limits (from your own Limits page), what to probe, how the stand-in differs |
| Orchestrator, limiter, retries, cache | Every number in `policy.yaml` and why; what is rejected vs delayed vs degraded |
| Load generator, aggregation, plots | Scenario multipliers; whether a scenario really stresses the system; the reading of results |
| Grading templates | **18 questions, gold keys, grades, regrade agreement** |
| Prose polish | The claims, the "number we could not have guessed", Vaani feedback, what you learned |

## 4. Where to look for AI mistakes (use this to review each Codex output)

Report only mistakes you actually find. This list is where they usually hide.

1. **Header semantics:** `x-ratelimit-*-requests` are per **day**; `x-ratelimit-*-tokens` are per **minute**. Code that treats the request headers as RPM is wrong.
2. **Closed-loop load generator** (awaits each response before sending the next): hides overload (coordinated omission). Arrivals must be scheduled by wall-clock plan.
3. **`time.time()` for durations** instead of `time.monotonic_ns()`.
4. **`first_token_ms` taken from the provider** rather than from the first user-visible byte after validation.
5. **Tokens missing** for failed attempts, repairs, verify calls; cached tokens not separated; reasoning tokens not budgeted.
6. **Hidden retries** in httpx transports or SDKs on top of the orchestrator's own.
7. **Verifier "pass" on missing/invalid output**; citations taken from model text instead of chunk metadata; arXiv version dropped.
8. **Percentiles over all rows** (including failures), or p99 reported without n.
9. **Limiter races:** check-then-reserve split across an `await`.
10. **Stand-in that always returns valid JSON instantly**, or "calibrated" from means; fixed latency; unseeded randomness.
11. **Tests that call the real API**, or log the Authorization header.
12. **Cache key** missing model, prompt version, corpus hash, or setting.
13. **Daily reset assumed to be local midnight.**
14. Mixing **scaled** stand-in rates with real limits in a table without labelling.

## 5. Likely live-interview changes (rehearse one-line edits)

- "Halve the deadline." (`deadline_ms`) What happens to rejections and to p95?
- "Disable retries." / "Allow 3 retries." (`retry.max_per_request`) Re-run S7.
- "Swap in setting B." (`--setting B`)
- "Prioritise requests that already have a draft over new ones."
- "What if the limit drops to a quarter?" (`/admin/limits`)
- "Show me why request X was slow." (`make timeline`)
- "Why not stream the draft immediately?" (D2; know the measured cost)

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Free TPD too small for 54 eval runs | high | high | verifier compact, effort low, k small, spread over UTC days, ledger pre-check, reruns reserve |
| `gpt-oss-20b` unavailable on your account | low | medium | `GROQ_MODEL` env; confirm on Models page tonight |
| JSON schema + streaming incompatible | medium | medium | fallback `json_object` or non-stream draft; record |
| Accidental real calls burn budget (Codex tests) | medium | high | guard in `05`; tests default to stand-in |
| Stand-in too easy/too hard | medium | medium | capacity probe; adjust multipliers; document |
| p99 unreliable | certain | low | state n; do not over-interpret |
| Grading takes longer than planned | medium | medium | 18 questions only; short gold keys |
| Write-up exceeds 2 pages | high | medium | word budgets; table font 7.3 pt; drop extra scenario rows |
| Key leaked in history exports | low | high | grep before upload; rotate the key after submission |
| Time lost on cross-encoder install | medium | low | RRF-only fallback |
