# 05. Deployment and runbook

"Deployment" here means: reproducible local setup for you, the reviewer, and the live interview. No cloud, no Docker (listed as not built).

## 1. Prerequisites

- Python 3.11 or 3.12, `uv` (or `pip` + venv), git
- A Groq free account and API key (console.groq.com). **One account only.**
- ~1 GB disk (PDFs, ONNX models for fastembed)

## 2. Groq account steps (do these tonight, before any code)

1. Create the key; store as `GROQ_API_KEY` in `.env` (gitignored). Never paste it into a chat with an AI or into the repo.
2. Open **Settings -> Limits**. Screenshot it, with the date and the model row for `openai/gpt-oss-20b`. Hover the TPM value to see whether there is an "X in / Y out" split (ITPM/OTPM). Save as `results/limits_page_<date>.png` (check the screenshot shows no key or email you do not want to share).
3. Open the Models page; confirm `openai/gpt-oss-20b` is available to your account and note the date. If not, pick another free text model that supports streaming and structured output and set `GROQ_MODEL`.
4. Note current list prices for the model into `config/prices.yaml` with the date read.
5. Check the Usage/Dashboard page and write down tokens already used today (UTC day), because the daily budget is shared with everything you run.

## 3. Setup

```bash
git clone <your repo> relay && cd relay
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # pyproject pins versions after first successful install
cp .env.example .env                # fill GROQ_API_KEY; leave RELAY_ALLOW_REAL=0 for now
make ingest                         # download 3 PDFs, assert versions, build chunks A/B/C + indexes
make test                           # all tests run against the stand-in only
```

`.env.example`:
```
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
STANDIN_BASE_URL=http://127.0.0.1:8081/openai/v1
RELAY_PROVIDER=standin          # standin | groq. Default must be standin
RELAY_ALLOW_REAL=0              # real calls require this = 1 AND --real on the CLI
RELAY_REAL_TOKEN_CAP=60000      # hard per-session cap on real tokens
RELAY_REAL_REQUEST_CAP=150
```

### The real-call guard (important)
The Groq daily limit is shared by everything you run, including Codex running tests. Therefore: default provider is the stand-in; the Groq adapter refuses to send unless `RELAY_ALLOW_REAL=1` **and** the process was started with `--real`; every real call is written to the SQLite ledger and the session aborts at `RELAY_REAL_TOKEN_CAP`/`RELAY_REAL_REQUEST_CAP`. Tests must never import a real client with a real key.

## 4. Commands (Makefile targets)

| Command | Does |
|---------|------|
| `make ingest` | download/verify PDFs, build chunk files and indexes |
| `make retr-eval` | offline gold-hit@k / MRR for settings A-D (zero tokens) |
| `make standin` | start the stand-in on :8081 with a profile (`PROFILE=scaled`) |
| `make relay` | start the Relay API on :8080 (`RELAY_PROVIDER=standin\|groq`) |
| `make probe` | `limit_probe.py --real` (asks for confirmation; prints token estimate first) |
| `make eval-real SETTING=A` | run the 18 questions sequentially on Groq, setting A/B/C |
| `make calibrate` | fit stand-in calibration from `results/events.jsonl` |
| `make capacity` | find S on the scaled profile |
| `make scenario NAME=burst` | run one stand-in scenario (S1..S8) |
| `make scenarios` | run all stand-in scenarios, seeded |
| `make aggregate` | build `results/results.csv` and `results/tables.md` |
| `make timeline RID=...` | print waiting/working/retrying split for one request |
| `make check` | pre-submission checks (below) |

## 5. Running order on the real service (token-aware)

See `07` for the budget per UTC day. In order:

1. `make probe` (tiny calls) -> Table 1 raw data.
2. `make eval-real SETTING=A`, then `B`, then `C`, on separate days if the TPD budget requires. The runner pre-checks the ledger and refuses to start a run that would exceed 70% of the remaining TPD.
3. `make calibrate`.
4. Real burst of 6 (`loadgen real-burst --n 6 --real`).
5. Everything else runs on the stand-in with no real tokens.

## 6. Reproducibility notes for the interview

- `config/policy.yaml` holds every policy number; a live "change X" request should be a one-line diff plus `make scenario NAME=...`.
- Every run writes `run_manifest.json` with seed, profile, policy snapshot and git commit. Commit before each run so `git commit` in the manifest is meaningful.
- Keep a `NOTES.md` with timestamps of what you ran and why; it makes the live conversation much easier.

## 7. Submission folder (email E4-E7)

```
relay-submission/
  writeup.pdf                      # max 2 pages
  results.csv
  code/                            # repo without .env, .venv, data/indexes, model caches
  raw/ events.jsonl  limit_probe.jsonl  run_manifest.json  grades.csv  limits_page_<date>.png
  conversation_history/
    codex_session_*/               # real exports or pasted transcripts, uncleaned
    chatgpt_planning_*.md          # the earlier analysis
    claude_planning_*.md           # this planning chat
  README.md                        # 10 lines: how to run, where each table number comes from
```

### `make check` must verify
- `results.csv` has the 12 required columns first, in order; `status`, `source`, `answer_grade` values valid; no duplicate `request_id`.
- Every number in `results/tables.md` equals the value recomputed from `results.csv` (that is how reviewers will check), and the write-up's tables are pasted from this file.
- No secrets: grep the whole folder for `gsk_`, `GROQ_API_KEY=` with a value, `Authorization`, and your email; fail if found (including in the conversation history).
- Write-up PDF is <= 2 pages and contains all 12 headings in order (script: extract text, check heading order).
- Folder opens in a **private/incognito window** with "Anyone with the link can view".

### Drive steps
1. Upload `relay-submission/` to Drive. 2. Share -> General access -> "Anyone with the link" -> Viewer. 3. Paste the folder link into the form from your original email, using **the same email address you applied with**. 4. Wait for the confirmation email; check spam; screenshot it. 5. Do not edit the folder after the deadline.

## 8. Failure playbook (what to do when something breaks)

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| 429 right away on a fresh day | Earlier runs used today's TPD, or reset time differs from assumption | Read `x-ratelimit-remaining-*`; check the dashboard; wait or move the run to the next UTC day; do not use another account |
| Draft JSON invalid | structured output not honoured with streaming | fall back to `json_object` + Pydantic, or non-stream for the draft; record the change |
| `cached_tokens` always 0 | prefix under model minimum, or prefix not byte-identical | make the system prompt longer/static; check no variable text before it |
| Stand-in results too clean | fault probs 0 or capacity too high | raise load multiplier; confirm fault config in manifest |
| Percentiles look odd | mixing failures into latency, or tiny n | recompute over `ok` only; print n |
| Cross-encoder install problem | ONNX/runtime mismatch | fall back to RRF only; label setting C "no rerank" honestly |
