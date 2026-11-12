# P1 implementation notes

All supplied files (`README.md`, `AGENTS.md`, `prompt.md`, and the eight files
under `docs/`) are preserved byte for byte. `P1_FILES.txt` lists the additions.
The integrity check caught Ruff formatting Python code fences in the architecture
Markdown; its original bytes were restored and matched the original SHA-256.
Markdown is now excluded from Ruff to prevent that recurring.
The architecture's later-prompt modules are docstring-only placeholders.
Later Make targets fail explicitly with exit code 2.

## Setup and commands

Run from the repository root:

```text
uv sync --extra dev --locked
make ingest
make test
uv run --locked python -m ruff check .
make check
make retr-eval EVAL_ARGS=--test-fixtures
make retr-eval
```

The final command intentionally refuses an empty actual dataset. The user must
write and verify the 18 questions and gold keys. Synthetic examples live only
under `tests/fixtures/`; fixture metrics are not paper-quality measurements.
There is no LLM call in P1.

`make ingest` explicitly prepares the dense C/D index after chunking, outside
the per-query deadline. Use `make ingest INGEST_ARGS=` for lexical-only ingestion.
An initial cold hybrid smoke check exceeded its 120-second query deadline while
building vectors. Batch preparation and configured ONNX threads/batch size avoid
putting that unbounded setup work in an ordinary request. These CPU settings are
engineering assumptions, not calibration measurements.

Python 3.12.10 was used. Direct dependencies are pinned in `pyproject.toml` and
transitive dependencies in `uv.lock`. NumPy has separate pins for Python 3.11
and 3.12+. Python 3.11 is dependency-resolved but has not been runtime-tested.
GNU Make 3.81 was installed using winget (`GnuWin32.Make`). In an already-open
PowerShell session the executable may need its installed directory in PATH:

```powershell
$env:Path = 'C:\Program Files (x86)\GnuWin32\bin;' + $env:Path
```

`PYTHON` can override the Make interpreter. No mypy configuration is supplied.
The `check` target explicitly identifies its P1 scope; submission verification
belongs to P7.

## Corpus and extraction review

All three page-one version stamps were verified against the exact configured
URLs: Transformer v7 (15 PDF pages), Moshi v1 (66), Dynamic Chunking v2 (39).
The manifest stores actual download SHA-256 values, fetch timestamps and a
corpus hash over the ordered document hashes. Cache reuse revalidates stamps
and hashes, and preserves original fetch timestamps. A hash mismatch aborts.
Progress is stored separately so an incomplete download never advertises a
complete corpus hash.

Poppler-rendered first pages and the requested passages were visually reviewed;
intermediate renders are under ignored `results/pdf-inspection/`. Review covered
Transformer PDF pages 5-6 (section 3.2.3 and Table 1), Moshi's first-page latency
qualification, and Dynamic Chunking PDF pages 4, 6 and 7 (architecture, routing,
and dechunking). No OCR was needed.

Observed extraction issues and limits:

- Transformer Table 1 is borderless. Default ruled-table detection missed it;
  caption-bounded text alignment now also detects table cells. Table rows retain
  extracted line breaks rather than inventing corrected mathematical notation.
- Equations/superscripts flatten, and some mathematical glyphs become control
  characters, particularly in Dynamic Chunking's routing equations. Nearby prose
  and section paths remain readable. Do not use this extraction as a typeset
  equation parser. The diagnostics flag damaged text and table inspection needs.
- Floating tables inherit the section active at their physical placement; this
  can differ from the section discussing them. Use PDF pages and table labels
  when checking a human gold key. First-page material may have section `null`.
- Paragraph windows stop at page/section/kind boundaries. Final partial chunks
  are allowed; non-final windows prefer paragraph boundaries within the configured
  tolerance. There is no invented padding. Page references are one-based PDF pages.
- Layout classification is heuristic. `data/extraction_diagnostics.json` reports
  column counts, table detection, warnings and inspection passages.

`o200k_base` is only an approximation of provider token usage. Its upstream BPE
asset is bundled with SHA-256 validation and license attribution so cold tests
never fetch tokenizer data. Embedding models are not bundled or loaded at import.

## Retrieval and safety

A/B use BM25Okapi. C/D use the specified BGE model, cosine ranks and RRF. C
reranks up to 12 candidates to four; D skips reranking and uses identical chunks.
The installed fastembed 0.9.1 API was inspected: `TextEmbedding.query_embed`,
`passage_embed`, and `TextCrossEncoder.rerank` are supported. Both configured
models downloaded and completed an inference check successfully, including the
supported `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder.

Original BM25/dense scores and ranks, RRF scores, raw rerank scores, and source
metadata accompany each hit. Negative BM25 corpus scores are shifted before
hinting; cross-encoder logits are mapped through sigmoid before a hint boost,
so multiplying by 1.3 never penalises a hinted paper. Ties use chunk IDs.
Non-positive cosine similarities do not contribute a dense rank; the configured
zero floor is an assumed neutral-signal policy, not a calibrated relevance cutoff.
An exact paper title or `in/according to/from the ... paper` permits a hard
filter; ambiguous keyword guesses only boost. Callers can pass `paper_filter`
for other explicit names. Multi-paper comparisons are not implicitly restricted.

Numeric retrieval appends immediate previous/next chunks of the top hit, within
the same paper/section/kind. These can exceed the normal base result count.
Cross-encoder failure explicitly reports `rerank_method=none` with a diagnostic.
Dense-model failure returns `error`, never a silent model substitution.
Timeout outcomes may leave the underlying CPU worker finishing; model work is
serialised within each engine. No retry or provider client is implemented.

The guard requires the environment permission and the literal process `--real`
flag, and atomically reserves worst-case session requests/tokens. It does not
refund reservations. P2 must integrate actual usage and the durable SQLite
ledger before any real adapter exists. No Groq calls were made.

## Assumptions and specification issues before P2

No supplied document was edited. Suggested clarifications:

1. `prompt.md` P1 asks for FAKE rows in `eval/gold.jsonl`, while the current user
   instruction confines them to test fixtures. The latter is implemented; actual
   question/gold files are empty.
2. The older acceptance text demands target +/-15% without a partial-chunk
   exception. The current instruction explicitly permits short final chunks.
3. Architecture section 4.1 omits diagnostics/rerank fields required by P1. They
   are additive fields; all documented public fields are retained.
4. `AGENTS.md`/architecture say wall-clock only for `start_time`, while the
   ingestion spec also requires `fetched_at`. Wall-clock is used for required
   fetch provenance only; all measured durations use the injected monotonic clock.
5. Design D8 says to overwrite the local minute-token estimate with the header;
   architecture section 3.2 instead says `min(local_estimate, header_remaining)`.
   This needs a P2/P5 decision before limiter implementation; P1 has no limiter.

The cross-encoder is deliberately a supported model selected after inspecting
the installed API; the documents do not pin a cross-encoder name. RRF constant,
timeouts, geometry/font thresholds and short-block duplication threshold are
labelled assumed engineering choices in `config/retrieval.yaml`, not measured
or calibrated values. Real answer quality remains unmeasured until the user
provides the human-authored evaluation set.

## Executed verification

- Dependency installation and locked sync succeeded on Python 3.12.10.
- Final complete pytest suite through `make test`: 58 passed, no network access.
- `ruff check .` passed; formatting checks passed with Markdown excluded.
- `make retr-eval EVAL_ARGS=--test-fixtures` completed A-D using synthetic data.
- `make retr-eval` refused the empty actual dataset with an actionable message
  and exit code 2, as intended.
- Actual corpus validation passed: A 665, B 467, C 532, D 532 chunks; maximum
  tokens 256/512/384/384; unique IDs, valid pages, C/D byte identity, Table 1
  classification, Moshi latency qualification and methodology section tracking.
- Source-document SHA-256 comparison: all 11 supplied files match the originals.
- Staged-file inspection found no credentials or generated PDF/model/index files.
- `make ingest` completed including the C/D dense index: 532 x 384 float32 values,
  all finite. The generated index remains excluded from Git.
- Actual-corpus A/B/D/C availability checks all returned `ok`; C reported
  `rerank_method=cross_encoder`. Results are in ignored `results/retrieval_smoke.json`.
  This is not human answer-quality grading.
- The first C availability call took 10,235 ms including model initialization,
  exceeding the future application's assumed 8-second deadline. Warm-process
  latency is not measured. Before P4/P5, measure startup/warm costs and decide
  how model preparation and the deadline policy fit together; no deadline claim
  is made for the as-yet-unimplemented application.

# P2 decisions (12 November 2026)

P2 extends the existing P1 commit; no supplied document is edited. Real Groq
execution is prohibited during this implementation, including both probe scripts.

For P5, select conservative reconciliation: retain local usage and outstanding
reservations, timestamp provider observations with the monotonic clock, and use
the smaller of the local available capacity and the latest valid provider
snapshot. Older/out-of-order observations cannot increase capacity. Reconcile
reservations with actual usage, and allow window recovery only through modelled
expiry/reset. P2 captures observations; it does not implement the P5 limiter.

Unknown/malformed 429 evidence maps to `rate_limit_unknown`; Retry-After alone
does not establish minute versus daily scope. Runtime classification patterns
remain empty until explicitly supplied from actual evidence. Test fixtures are
hypothetical, not claims about Groq's error messages.

The [Groq Structured Outputs documentation](https://console.groq.com/docs/structured-outputs)
was checked on 12 November 2026: streaming is not supported with Structured
Outputs. Production paths are non-streaming JSON Schema and streaming JSON Object
with application-side validation. A schema/stream experiment remains explicitly
opt-in for the probe. An explicit pre-dispatch compatibility conversion may
choose JSON Object for this documented combination; arbitrary failures never
cause an automatic downgrade or another HTTP attempt.

The [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits)
identifies request headers as RPD and token headers as TPM. Additional headers
are retained without inventing meanings. Account-specific quotas and actual
response shapes remain unverified; P2 does not measure them.

## P2 implementation and verification

The clock protocol provides `monotonic_ns()` and UTC-aware `utc_now()`.
`SystemClock` uses the corresponding system clocks; `FakeClock` advances only
when instructed and permits independent UTC jumps. Provider durations, content
TTFT, reasoning-delta timestamps, event offsets and cooldowns use monotonic time.
UTC wall-clock is used for persisted ledger/probe provenance, as P2 explicitly
requires, rather than elapsed-time arithmetic.

Events retain the architecture's `rid`, `run`, `t_ms`, `ev` envelope and
`usage.in/out/cached` and `hdr.rem_tok/rem_req/reset_tok` fields. Additional safe
headers and timestamped quota observations are additive. One locked JSON append
prevents interleaving; persistence failures raise `EventWriteError`. Recursive
redaction removes authorization fields, masks sensitive values and registered
credentials, rejects raw environment mappings, and bounds response/error bodies
to 500 characters, including structured bodies. Probe JSON and JSONL use the
same sanitizer; raw outgoing paper/message bodies are not logged.

SQLite retains all ten required `calls` columns and adds session/request IDs,
status/error details, total/countable usage and reservations. WAL, explicit
transactions and indexes support atomic inserts and model/UTC-day/session
queries. Each dispatched attempt has its own durable record, including 429s;
this is local attempt accounting, not a claim about provider quota consumption.
Known countable usage is input minus cached plus output; reasoning tokens are
never added again. Missing raw usage stays NULL and is reported as unknown.
Pending/unknown attempts conservatively retain their reservation, increased by
larger known partial/total evidence. Inconsistent historical usage fails closed.
Known usage reconciles the reservation without deleting the attempted request.

The provider uses raw `httpx.AsyncClient`, four explicit timeouts, zero transport
retries, no redirects and no proxy environment. It parses optional usage/timing
fields defensively and preserves status and safe request IDs. Malformed output,
JSON validation, HTTP errors, timeouts, disconnects, cancellation and persistence
failures have typed outcomes. Known usage survives output-validation failure.
Non-streaming TTFT remains unavailable. The SSE parser supports keep-alives,
multiline data, usage-only/final usage chunks and `[DONE]`; reasoning and content
times are separate, and only actual content determines TTFT. Cancellation closes
the response and settles accounting before propagating cancellation. Internal
delta callbacks are intended for future orchestrator buffering; no client-facing
application or unvalidated-response delivery is implemented in P2.

External requests require both permission gates, an allowed HTTPS origin, a
credential, a durable ledger and an atomic positive token reservation. Guards
share a default process session across instances; outstanding calls count before
I/O, preventing concurrent cap bypass. Local literal-loopback requests use no
real credential and are marked `real=false`. No provider SDK, hidden retry,
P3 stand-in service, P4 orchestrator or P5 limiter was added. The localhost fake
HTTP server exists only in tests and uses ephemeral registered ports.

Both probe entrypoints default to planning only, before credential lookup,
provider-client construction or opening a ledger. Execution requires `--local`
or both real permission gates and explicit confirmation (`RUN`/`--confirm RUN`).
Real limit experiments additionally require supplied account limits and verified
remaining-day headroom. All phases share the guarded session, stop on daily
evidence/cap/persistence failures, and honor the longest outstanding Retry-After
deadline. An excessive cooldown aborts instead of sending early. Recovery is
the first observed HTTP 2xx after a 429, separately from JSON validity.

Executed on Python 3.12.10 with the existing locked dependencies:

- Complete suite through `make test`: **142 passed** (including all 58 P1 tests).
- `ruff check .`: passed. `ruff format --check .`: 79 files already formatted.
- Mypy is not configured; no mypy result is claimed.
- `python -m scripts.probe_groq --dry-run --experiment-stream-schema`: at most
  3 requests and 648 estimated tokens; no client/ledger/network opened.
- `python -m scripts.limit_probe --dry-run`: at most 56 requests and 15,690
  estimated tokens: RPM 40/4,800, TPM 3/9,330, concurrency 8/960, recovery 5/600.
  The TPM phase remains below the specified 10K estimated-token ceiling.
- `make probe` produced the same limit plan; `make probe-shapes` planned the
  two default shapes (2 requests, 432 estimated tokens). Both opened no client,
  ledger or network connection.
- Probe execution, including subprocess entrypoints, passed against the fake
  localhost service. Tests cover cancellation/connection closure, malformed and
  incomplete streams, unknown usage, header interpretation, 429 fixtures,
  cooldowns, parallel reservations, default refusal, no hidden retry, sanitizing
  artifacts and failed persistence. External-origin guard tests use mock HTTP
  transports and synthetic credentials, never the Groq service.
- No real Groq API request was sent. No real shape/limit result is claimed.
- Original 11 supplied documents are unchanged. Complete P2 source inventory is
  in `P2_FILES.txt`; generated databases, probe outputs and caches stay untracked.

Timeouts, offered rates, tokenizer overhead and probe safety fractions are
explicit assumed engineering settings in YAML. Local token estimates are never
presented as actual provider usage; unknown account limits/headroom remain NULL.
Production error-pattern configuration is deliberately empty until observed
evidence exists. Real account quotas, streaming response/usage shapes, actual
model behavior and latency remain unmeasured. The optional fake-server schema
streaming experiment proves tooling/parsing only, not Groq compatibility.

Suggested original-document updates (not applied): align design D8 with the
selected conservative P5 policy; describe unknown 429 outcomes and nullable
usage; state the schema/streaming split and explicit compatibility policy; and
update the runbook to explain that `make probe` and `make probe-shapes` now
default to dry runs. Existing P1 notes above are historical, including the guard's
former non-refunding behavior; P2's durable reconciliation supersedes it.

# P3 integration checklist and decisions

The clean starting point is P2 `c4b67f08857d2d2391daee1dc7e748f4ac70ec10`
on main, with the expected origin. All original documents remain immutable.

- P2 appends `chat/completions` to a base URL ending in `/openai/v1/`.
  Literal loopback is local, needs no key/real permission/reservation, and ledger
  attempts are `real=false`. No second provider adapter is needed.
- Non-stream responses expose `choices[0].message.content`, IDs/model and usage.
  SSE carries `choices[].delta.content`, optional reasoning, top-level usage or
  `x_groq.usage`, empty-choice usage snapshots and `[DONE]`.
- Usage includes input/output/cached/total separately; reasoning is a subset of
  output. The existing offline `get_encoding().encode()` counts simulated tokens.
- Retry-After is numeric seconds; request headers mean RPD and token headers TPM.
  Unknown simulator 429 codes remain unknown to the production classifier.
- Calibration joins `call_start/first_delta/call_end` by call_id. Kind/model/source
  are on start/end; first_delta carries `what` and `provider_elapsed_ms`, end
  carries total_ms and usage.in/out/cached plus completion_time when available.
- Clock and EventWriter are constructor injections. Service factories also inject
  a dedicated RNG and async sleep primitive for deterministic tests.

Current P3 instructions supersede the older demand to copy observed Groq 429
shapes: none have been observed, so error codes/reset formatting are explicitly
simulated. They also supersede the older scaled default: development is synthetic;
free_exact/scaled refuse absent account values. Provider/header policy is unchanged.
UTC wall-clock is required for UTC quota boundaries and wire `created` provenance;
all service delays/windows/durations use the monotonic clock. Internal stand-in
streaming is provider simulation, not P4's user-visible answer release.
