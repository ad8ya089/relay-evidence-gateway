# AGENTS.md: rules for AI coding assistants working on Relay

Project: a paper-grounded QA pipeline with an admission-controlled control plane, a Groq adapter, and a stand-in model service, plus load tests. Specs live in `docs/` (read `docs/01_DESIGN.md` and `docs/02_ARCHITECTURE.md` before coding). If code and docs disagree, stop and say so; do not silently pick one.

## Hard rules
1. **Never make real network calls to Groq in tests or by default.** Default provider is the stand-in. Real calls need `RELAY_ALLOW_REAL=1` **and** the `--real` CLI flag, are written to the SQLite ledger, and abort at `RELAY_REAL_TOKEN_CAP` / `RELAY_REAL_REQUEST_CAP`.
2. **Never print, log, or store the API key or Authorization header.** Redact in events and probe logs. Truncate bodies to 500 chars.
3. **No hidden retries.** httpx client has zero transport retries; no SDK retry. Only `Orchestrator` retries.
4. **Durations use `time.monotonic_ns()`.** Wall-clock only for the `start_time` field.
5. **Do not invent facts**: no made-up Groq limits, 429 body shapes, model capabilities, paper section numbers, page numbers, gold answers, or "calibrated" values. Unknowns are config values labelled `assumed`, or `not measured`.
6. **Citations are rendered from chunk metadata**, never taken from model text. Every citation carries `arxiv_id` and `arxiv_version`.
7. **A missing or malformed verifier result is not a pass.**
8. **Nothing is sent to the client before validation completes** (admission rejections excepted).
9. Do not add LangChain, LangGraph, Redis, Kafka, hosted vector DBs, ORMs, or a frontend.
10. Do not refactor files outside the scope of the current prompt. Do not change `docs/`; propose edits in your final message instead.

## Conventions
- Python 3.11+, type hints everywhere, Pydantic v2 models for all cross-module data, `asyncio`, `httpx.AsyncClient`, `pytest` + `pytest-asyncio`.
- Config in `config/*.yaml`, loaded once into typed settings; no magic numbers in code.
- Inject the clock (`relay/obs/clock.py`) and RNG so tests are deterministic.
- Structured events via `relay/obs/events.py` only (JSONL, schema in `docs/02_ARCHITECTURE.md` section 5).
- Small functions, explicit return types, explicit typed outcomes for every failure (no bare `except`).
- After each task run `make test` and `ruff check` (and `mypy` if configured); paste the summary in your final message.

## Definition of done for any prompt
- Acceptance tests named in the prompt exist and pass.
- You list: files created/changed, commands to run, anything you assumed, anything you could not verify.
- You do not claim something works unless you ran it.
