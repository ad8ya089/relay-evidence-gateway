# 00. Requirements traceability

Sources: the 4-page PDF brief ("B") and the email ("E"). Tick each box before submitting. Read this file twice, once now and once the night before you submit.

## A. Logistics (email)

| ID | Requirement | Where satisfied |
|----|-------------|-----------------|
| E1 | Deadline **Sat 14 Nov 2026, 9:38 AM IST** | `07` plan targets Fri 11 PM IST |
| E2 | Work alone | Do not share code or prompts with anyone |
| E3 | Any AI tool allowed; they judge how you direct it and what is yours | `06` section "How I worked with the assistant"; `07` checklist |
| E4 | One Google Drive folder with write-up, code, `results.csv` | `05` section "Submission folder" |
| E5 | Folder set to "Anyone with the link can view" | `05`; open it in a private window to test |
| E6 | Paste link into the form using the **same email you applied with** | The form link is in your original email (it appears blank in the pasted text); find it now |
| E7 | Confirmation email expected within minutes; check spam | Keep a screenshot of it |

## B. What to build (brief p.1)

| ID | Requirement | Where satisfied |
|----|-------------|-----------------|
| B1 | Python, C++ or Go | Python 3.11+ |
| B2 | At least **4 steps**, at least **1 tool call**, at least **1 branch or loop**, one step retrieves | Pipeline: interpret, `search_papers` tool, evidence gate (branch + broaden loop), draft, validate, repair loop, release. See `02` |
| B3 | Library = exactly 3 papers at exactly these versions: 1706.03762 **v7**, 2410.00037 **v1**, 2507.07955 **v2** | `ingest.py` asserts the version stamp on page 1; `corpus_manifest.json` holds sha256 |
| B4 | Every answer cites where in the papers; if not in the papers, **say so** | Code-rendered citations; `insufficient_evidence` path |
| B5 | Real hosted model on **Groq free tier**, own account and key | `openai/gpt-oss-20b` (configurable) |
| B6 | Stand-in service: can be **slow, fail, refuse at a limit, slow under load, stream in pieces** | `04` |
| B7 | Calibrate stand-in from real measurements; say where it is least like the real thing | `04`, write-up section 2 |
| B8 | Do not put the key in the submission | `.env` gitignored; scan before zipping (`05`) |
| B9 | Stay within provider terms; no extra accounts or tricks | One account, one key. Using two different models for load balancing is also avoided |

## C. What to run (brief p.1)

| ID | Requirement | Where satisfied |
|----|-------------|-----------------|
| C1 | Modest number of real requests to learn latency spread, limits, and behaviour when a limit is hit | `limit_probe.py`, real eval runs, real burst of 6 |
| C2 | At least **3 load shapes** of your own design | 5 required scenarios + 2 extra on the stand-in (`04`) |
| C3 | Raw results for both real and stand-in kept | `results.csv` + `events.jsonl` + `limit_probe.jsonl` |

## D. Write-up (brief p.2): max 2 pages, these headings, this order

Order is fixed. Note the two retrieval headings go **after heading 5**.

| # | Heading (use verbatim) | Must contain |
|---|------------------------|--------------|
| 1 | What I built and why | Design; the one decision that shaped it most |
| 2 | The real service and my stand-in | Limits found and **how**; stand-in behaviour; calibration; where it differs most |
| 3 | Results | TTFT and total at median and slow end (say which percentiles and **how many requests**), failures, given-up, per load shape; real vs stand-in labelled; **every number in the tables**; **a number we could not have guessed**; answer quality on **>= 15 self-written questions**; tokens per answer |
| 4 | Where the time goes | One slow request split into waiting / working / retrying; before or after first token; show how you know |
| 5 | When a limit is hit | Behaviour at per-minute and per-day limit: who is delayed, dropped, weaker, and why |
| 5a | Retrieval: quality versus cost | >= 3 settings (piece size, number fetched, re-rank); quality, tokens/answer, TTFT; choose one and say why; what you do with retrieval when the limit is close |
| 5b | How this library will mislead a naive retriever | **Three** ways, each with an example question |
| 6 | Retries | Policy; one case where retrying made things worse |
| 7 | What the person sees | At 3 PM the daily budget is gone: the reply text |
| 8 | What I'm unsure about, what I left out, what I learned | Least certain assumption and what breaks; **two** things deliberately not built; **one** new thing taught in your own words |
| 9 | Your feedback on Vaani | A few lines: what stood out, what confused you, one thing to change |
| 10 | How I worked with the assistant | Real (uncleaned) history attached; half page: **three** most important prompts and why; **one** wrong thing caught and how; **one** decision where you overruled it; what is distinctly yours |

## E. Tables (brief p.3): every cell filled, "not measured" where unmeasured; ms for time; one decimal for percentages

| ID | Table | Rule |
|----|-------|------|
| T1 | Limits found in real service | Rows: RPM, TPM, RPD, TPD, requests in flight, what it returns when hit, time to recover. Columns: value, how found, confidence (high/medium/low) |
| T2a | Latency per scenario | Min rows: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted. Columns: real or stand-in, requests sent, duration (s), offered rate/min, first token p50/p95/p99, total p50/p95/p99 |
| T2b | Outcomes, same rows | Succeeded %, rate-limited %, timed out/given up %, retries/request, peak queue length, tokens/answer, cost per 1,000 answers |
| T3 | Retrieval settings (>= 3) | Questions asked, correct and cited %, wrong %, correctly refused %, tokens/answer, first token p50/p95, total p50/p95 |

Reviewers **recompute from `results.csv`**. `aggregate.py` is the only thing that produces table numbers, and the write-up pastes its output (never hand-edit).

## F. `results.csv` (brief p.4)

One file named exactly `results.csv`, one row per request. Required columns in this order: `request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`. Allowed values: `source` in {real, stand-in}; `status` in {ok, rate_limited, timeout, error}; `answer_grade` in {correct, wrong, refused_correctly, refused_wrongly}. Extras go **after** these 12 (see `04`). Rows that are not quality-graded (load tests) leave `answer_grade` empty; declare that in the write-up.

## G. Ground rules (brief p.4)

- [ ] Within 4 days; 6-8 hours of real work (not scored on hours)
- [ ] Max 2 pages of writing, plus code, raw results, conversation history
- [ ] Own free account; **no key in submission**
- [ ] Daily limit is shared by everything you run (including Codex tests): see the real-call guard in `AGENTS.md`
- [ ] Only submit what you can defend in the live conversation, where you will be asked to **change the system**. Keep every policy in `config/policy.yaml` so a change takes one line.
