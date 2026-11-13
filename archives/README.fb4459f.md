# The Relay: project pack

Everything you (and Codex) need to build the Vaani "Relay" assignment. Deadline: **Sat 14 Nov 2026, 9:38 AM IST**. Plan to submit by **Fri 13 Nov, 11 PM IST**; treat Saturday morning as buffer only.

## Reading order

| # | File | What it is |
|---|------|-----------|
| 0 | `docs/00_REQUIREMENTS_TRACEABILITY.md` | Every requirement from the PDF brief and the email, mapped to the artifact that satisfies it. Re-check before submitting. |
| 1 | `docs/01_DESIGN.md` | The decisions, the numbers, and the defence of each. |
| 2 | `docs/02_ARCHITECTURE.md` | Components, state machine, schemas, event log, repo layout. |
| 3 | `docs/03_RETRIEVAL_EVAL.md` | Ingestion, 3 LLM-graded retrieval settings + offline diagnostics, how to write your 18 questions. |
| 4 | `docs/04_STANDIN_LOADTEST.md` | Stand-in service, calibration, 7 scenarios, metric and CSV definitions. |
| 5 | `docs/05_DEPLOYMENT_RUNBOOK.md` | Setup, commands, Groq account steps, Drive submission. |
| 6 | `docs/06_WRITEUP_TEMPLATE.md` | The 2-page write-up skeleton with the brief's exact headings and tables. |
| 7 | `docs/07_PLAN_AND_RISKS.md` | Hour-by-hour plan, token budget across UTC days, risks. |
| - | `AGENTS.md` | Drop in the repo root; Codex/Cursor read it every run. |
| - | `prompt.md` | **Seven** build prompts for Codex, in order. |

## The one-paragraph design

An application-owned state machine (interpret, `search_papers` tool, evidence gate with broaden loop, draft, deterministic + LLM validation, at most one repair, release) sits behind a control plane that owns admission, deadlines, token reservations, retries and a cache. Answers are **buffered until validated**, then released with citations rendered by code from chunk metadata, so the model cannot forge a citation. When capacity runs out the system sheds at the door (fast, with Retry-After) instead of queueing or sending an unverified answer.

## What I changed versus the ChatGPT plan, and why

I kept its skeleton (Python/FastAPI/asyncio/httpx/SQLite, application-led orchestration, validated release, open-loop load, CSV-first). These are the changes:

| # | Change | Why |
|---|--------|-----|
| 1 | **Design around tokens, not requests.** | Groq's current docs list the free tier at 30 RPM / 1K RPD / **8K TPM** / **200K TPD** for `openai/gpt-oss-20b` and `-120b`. At ~2.8K tokens per answer that is ~2.9 answers/min and ~70 answers/day, so TPM/TPD bind long before RPM. The ChatGPT plan treated RPM as the headline. |
| 2 | Default model `openai/gpt-oss-20b`, `reasoning_effort=low`. | It is on the current free table and supports prompt caching. Reasoning tokens count as output, so effort must be low. The Llama models it quoted from tutorials are no longer in the docs table. Confirm on your Limits page. |
| 3 | **Static-prefix-first prompts + record `cached_tokens`.** | Groq docs: cached tokens do not count toward rate limits. A fixed system prompt and schema first, variable content last, buys real headroom. Measure it. |
| 4 | **Compact verifier.** Verifier sees only claims + quoted spans, never full chunks. | ~600 input tokens instead of ~2K. Same validation, a third of the budget. |
| 5 | **Quote-anchored evidence.** Every claim cites `chunk_id` + a verbatim quote; code checks the quote is a substring of the chunk, and renders the citation from metadata. | Cheap, deterministic, kills fabricated citations before any model call. |
| 6 | **No partial answers.** Full validated answer, or an explicit "could not verify". | A half-answer that silently drops a claim is an embarrassing-answer risk. |
| 7 | **Answer cache + single-flight coalescing.** | The brief says bursts come "right after something goes wrong for many users at once", so many near-identical questions. Cache hits cost zero tokens and are still served when the daily budget is gone. Disabled in the main scenarios and shown in one extra scenario. |
| 8 | **Header reconciliation.** After every response the limiter overwrites its local estimate with `x-ratelimit-remaining-tokens`. | Limits "can change without notice". Headers are ground truth; local counters drift. |
| 9 | **Predictive admission**: reject at the door if the projected wait exceeds the budget, instead of FIFO-then-timeout. | Never spend tokens on work that will miss its deadline. |
| 10 | **Stand-in uses "scaled" limits** (free-tier ratios x30) for heavy scenarios, plus an exact-limit fidelity run against the real service; one scenario changes limits mid-run without telling the system. | Exact free limits would make a 200-request run last over an hour. Scaling every limit by the same factor keeps *which limit binds* identical. |
| 11 | **Dropped the optional planner LLM call.** Interpretation is local only. | A serial call costing ~1K tokens at 8K TPM is not affordable. |
| 12 | **Free retrieval diagnostics.** Offline gold-hit@k for four configs at zero tokens; only three are LLM-graded. | Token budget cannot afford a 4th LLM-graded setting. |
| 13 | fastembed (ONNX) instead of sentence-transformers/torch. | Smaller install, faster CPU, same function. |
| 14 | CSV keeps the 12 required columns first, in the brief's order; extras appended; run-level config lives in `run_manifest.json`. | Brief says every table number must be computable from the file; extras are needed for peak queue length etc. |
| 15 | **You write the 18 questions and gold answers.** AI does not. | The brief says "questions you wrote yourself" and the interview probes it. |
| 16 | Corrected the timeline. | Deadline is Sat 09:38 IST. Groq daily limits reset at 00:00 UTC (05:30 IST) per a secondary source; verify. That gives you only three full budget days. See `07`. |

## Things I could not verify, so verify them first (Prompt 2 builds the probes)

1. Your account's actual limits on the Limits page (docs say "high level summary and there may be exceptions"). Also hover TPM for an "X in / Y out" split.
2. Whether `response_format: json_schema` works together with `stream: true` on `gpt-oss-20b`, and where `usage` appears in a streamed response.
3. The 429 body shape and `retry-after` for a TPM versus an RPM hit.
4. Whether failed (429) requests consume tokens or requests against your quotas.
5. The daily reset time (secondary sources say 00:00 UTC; the official page did not state it in what I fetched). Capture `x-ratelimit-reset-requests` at the start of a day.
6. Current list prices for the model (for the "cost per 1,000 answers" column). Record the date you read them.

## Two honesty notes

- **Vaani feedback is yours.** I looked at the homepage: it shows "Lowest latency: best-in-class p90 latency of ~450ms" and a "Sub-500ms multilingual engine" stat, plus Teacher AI, Memories and call-screening features. Visit it yourself and write what you honestly think. My notes are in `06`.
- **AI-disclosure.** The brief wants your *real* conversation history. That includes this planning phase (ChatGPT and this chat). Export them, do not clean them up, and do not claim you caught an AI mistake you did not catch. `07` has a "where to look for AI mistakes" checklist so the real ones you find are easy to spot.
