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

## P3 component responsibilities and assumptions

`profile.py` loads cached Pydantic settings with numeric/distribution provenance.
`faults.py` owns a dedicated locked RNG, resettable state and public bounded
decision diagnostics. `generation.py` owns marker detection, chunk parsing and
the draft/verifier JSON contracts. `cache.py` owns the bounded hashed static
prefix cache. `quota.py` owns atomic per-model accounting. `simulation.py` joins
these components into a completion lifetime. `service.py`, `streaming.py` and
`admin.py` own HTTP, SSE and loopback-only administration. No P2 source or
contract changed, and no P4/P5 orchestration/admission implementation was added.

The production `app` and `create_app` factory are available. Factories accept a
profile, clock, RNG and async sleep primitive. The CLI runs one Uvicorn worker,
binds to 127.0.0.1 and disables forwarded-header trust. Admin snapshots contain
configuration, counters and sampled decisions, not messages or request headers.
Profile replacement/reset require quiescence (409 while requests are active);
limit updates can occur during service. A finishing non-stream response reports
the latest active limits after usage settlement, including a limit changed while
that request was running. Stream headers describe the reservation-time snapshot.
Call IDs remain unique across resets: a regression test first demonstrated that
reusing IDs let a late old finalizer settle a new reservation. RNG decisions and
quota/cache/statistics reset reproducibly; identity reuse is intentionally avoided.

Development limits and distributions are synthetic, with chosen/assumed sources.
The model name is a configured identifier, not an account-availability claim.
`free_exact` keeps RPM/TPM/RPD/TPD NULL and refuses missing values. Exact inputs
must carry observed provenance and date before running. `scaled` requires that
base and multiplies all four values by the chosen K=30, with chosen capacity 8;
free_exact capacity 4 remains assumed. No online generic quota was substituted.

Minute quotas use rolling 60-second monotonic windows; daily quotas use UTC
midnight. Token admission includes uncached estimated input plus the requested
completion maximum and every outstanding reservation. Known simulated completion
usage reconciles the reservation; failed/cancelled calls retain a conservative
charge. Requests rejected before admission consume no simulated quota. Outstanding
token reservations survive their original window/day until settlement. A 429
lists all exceeded scopes and uses the longest known recovery time. If pending
work or an oversized request makes recovery unknown, Retry-After is omitted.
Request headers mean RPD and token headers TPM; RPM/TPD appear in admin state.
Reset strings are simulated decimal seconds plus `s`, not observed Groq syntax;
nominal window expiry alone does not release a still-pending reservation.

Timing samples and fault draws are reproducible for the same seed and operation
start order. Network arrival ordering under concurrent load is outside that
guarantee; decision sequence numbers expose the actual order. Both TTFT and
decode duration use the exact load multiplier. Decode duration is completion
tokens divided by sampled tokens/sec. Slow successes multiply both components.
Independent fault draws have explicit server/timeout/disconnect precedence;
slow-success applies to calls without a hard fault. Server faults return the
configured 500/503. Timeouts withhold a response for the configured interval,
monitor disconnects, and eventually return simulated 504 if the caller waits.
Truncated SSE omits finish/usage/DONE and is detected by P2 as a disconnect; it
models incomplete protocol delivery, not a particular TCP-reset mechanism.
There are no detached sleeping tasks or automatic retries.

Streams delay first content by sampled TTFT, concatenate to the same complete
JSON as non-streaming, emit configured nonempty content pieces (bounded by the
text length), finish reason, optional usage-only final chunk, then `[DONE]`.
Cleanup is idempotent across generator/background finalizers and cancellation.
Usage input is estimated with P1's offline tokenizer and explicit message framing.
Output counters sample a synthetic volume, bounded by the requested maximum and
at least the JSON's visible token length. They can exceed literal JSON tokens;
they are simulated counters, not provider measurements. Reasoning is a subset
of output, cached tokens a subset of input, and totals are input plus output.
A completion cap too small for complete JSON returns 400 instead of partial JSON.

Drafts use the top usable multiline `<<chunk:ID>>...<<end>>` block and an exact
sentence span capped at 40 words; empty evidence explicitly refuses. IDs and
citations are never invented. Verifier input is exactly the architecture's JSON
list of `{i, claim, quotes}` objects; blank/malformed inputs fail explicitly.
Ordinary supplied claims receive simulator-supported verdicts; verify_fail_prob
marks one eligible claim unsupported. This exercises P4 control flow and does
not constitute semantic verification of scientific claims.

The cache hashes model plus exact initial system/developer messages. Dynamic
user questions/chunks are outside that prefix. TTL defaults to the assumed
7,200 seconds and is refreshed after completed generation; size/threshold are
configured. First use, different models, changed prefixes and expired entries
miss. Failed generation does not populate the cache; reset clears it. P4 must
keep dynamic material in user messages as specified by the architecture.

## P3 calibration engine

`make calibrate` consumes the P2 JSONL envelope. Start/end identity and source
must match, calls join by run/call_id, models are isolated, and only real-labelled
draft/verify attempts qualify. Stand-in events, synthetic/test flags/identifiers,
fixture paths, duplicate/incomplete calls and failed metric observations are
excluded. First-delta events normally inherit start source; an explicitly
stand-in-labelled delta cannot contaminate a real-labelled call. Source labels
are not proof of authenticity: actual event provenance still needs human review.
The fixture-only in-memory extractor used by statistics tests has no CLI flag.

TTFT uses first content provider_elapsed_ms, never reasoning or fabricated
non-stream TTFT. Decode rate prefers reported completion_time, otherwise uses
completion minus first content; invalid durations/incomplete usage are excluded.
Below the configured 50-pair threshold, resampling preserves observed TTFT/rate/
output-token tuples, with explicitly chosen 5% conditional lognormal tail mass.
Larger samples fit lognormal parameters using p50/p95 and the specified 1.645
formula. Completion-token samples remain empirical. Paired complete observations
are compared with seeded simulations at p50/p95/p99, including ratios. Missing
metrics say unavailable. p99 warns below 100 observations and for >25% mismatch.

Alpha fitting requires explicitly declared genuine sequential and burst-six run
IDs, comparable call kinds and capacity <6; otherwise alpha=0.5 assumed. The
report preserves the capacity's provenance and full calibration configuration.
Observed nonzero error frequencies need a meaningful sample; zero observed
errors retain chosen simulation defaults instead of claiming zero true risk.
The development defaults are 2.5% server errors and 10% slow successes, chosen.
Timeout/disconnect/verification settings remain explicitly labelled.

Generated `config/calibration.json` is ignored by Git. No real observations were
available: the executed report is **uncalibrated**, with zero draft/verify samples.
Synthetic known-parameter tests validate fitting mathematics only. An explicit
`--calibration` overlay can apply fitted metrics without changing fallback YAML;
model identity is checked and other models retain fallback timing. Profile swaps
clear the overlay. No real-account limits or fitted data were committed.

## P3 execution and review before P4

Executed startup checks: development succeeded; free_exact and scaled exited 2
with the four missing Limits-page inputs. The default 127.0.0.1:8081 launch found
an existing listener and was not allowed to replace/terminate it. An independent
CLI launch on ephemeral 127.0.0.1:57930 completed SSE with status 200 and complete
usage using credential=None; that child was terminated after the smoke test.
Real localhost integration tests also exercise live Uvicorn on ephemeral ports.

Before P4, retain single-worker accounting, the exact verifier input layout and
static-prefix placement. Buffer all internal deltas until validation. Real quotas,
latency distributions, cached-token rules and failure shapes remain unverified.
No P2 defect required a provider change. Proposed original-document edits remain
unapplied: replace copied-real-429 assumptions with explicit simulation until
observations exist, document the development default/refusing exact profiles,
and describe quiescent profile/reset administration and reservation semantics.
All original 11 project documents are preserved. No real Groq calls were sent,
no credential was needed/read by the stand-in, and no real probe was executed.

Final P3 verification:

- Full pytest and `make test`: 212 tests passed, comprising the unchanged 142
  P1/P2 tests (including P1's original 58) and 70 P3 acceptance/integration tests.
- `ruff check .` passed; `ruff format --check .` reported 100 files formatted.
  Mypy is not configured.
- Provider compatibility includes non-stream and SSE draft/verify calls, final
  usage, cached metadata, content TTFT, simulator 429 classification and no key.
  Six concurrent HTTP operations with controlled sleeps exercise the exact load
  multiplier; live HTTP tests verify timeout/cancellation/disconnect cleanup.
- `make standin PROFILE=development STANDIN_ARGS=--check` succeeded. Exact/scaled
  checks refused missing account limits as intended. `make calibrate` completed
  uncalibrated with zero real samples and explicit unavailable/p99 warnings.
- Source-document SHA-256 mismatches: zero of 11. Existing relay sources/tests
  are unchanged. Staged inspections found no secrets or generated binary/data
  artifacts. The complete 31-file inventory is `P3_FILES.txt`.

Commands available: `make standin PROFILE=development` (default port 8081),
`make standin PROFILE=development STANDIN_ARGS="--port PORT"` when that port is
already occupied, and `make calibrate CALIBRATE_ARGS="--events PATH"`.
Calibration application is explicit via stand-in `--calibration PATH`; it never
silently rewrites chosen/assumed profile configuration.

# P4 integration contract (13 November 2026)

Starting main/origin main are P3 `0d25179b5e7e48e0a0cdc427050e787fb8f1666b`,
with a clean tree. Original documents and actual eval questions/gold are protected.

- P1 `RetrievalEngine.search_papers(query, paper_filter=None, top_k=None,
  setting="C", *, paper_hint=None, numeric=None)` is async and returns the existing
  `RetrievalResult`/`RetrievalHit` models. Metadata includes pinned IDs/versions,
  title, section (optional in P1), PDF pages, scores/ranks/method diagnostics.
  Numeric=True already appends previous/next same-section/kind chunks. There is
  no separate public neighbour-fetch API; P4 uses this existing capability.
  P1 work may continue in its CPU thread after cancellation; no duplicate worker
  is deliberately started after a timeout while that work could still be running.
- P2 `Provider.complete(LLMRequest, *, on_delta=None)` buffers results, returns
  nullable `Usage`, first-content/reasoning timings and headers, and raises typed
  provider errors/cancellation. `CallMetadata` carries shared request/run IDs,
  unique call IDs, kind and arrival_ns. JSON Object streaming and non-stream
  JSON Schema/Pydantic validation are already implemented. No provider retry.
  Ledger records are owned by P2; P4 reconciles by those call IDs without inserting
  duplicate rows. Unknown usage remains unknown, with known subtotals retained.
- P3 detects RELAY_DRAFT_V1 / RELAY_VERIFY_V1 in system/developer content. Draft
  chunks are multiline `<<chunk:ID>>...<<end>>`. Verifier user content is exactly
  the JSON list `{i, claim, quotes}`; no chunks/question accompany verification.
  Initial static messages form P3's exact model-aware cache prefix. Repairs use
  the same draft prefix. P3 is a deterministic plumbing simulator, not a semantic
  research-answer evaluator; no real quality or Groq behavior is inferred.
- P2 Clock/EventWriter remain the sole timing/event abstractions. P4 events
  contain IDs, counts, statuses and timings, never prompts, chunks or draft text.

Clarifications: local/model evidence refusals have status ok; validation
abstention has status error, per design section 5 and architecture. P4 explicitly
requires section/page/version provenance before release, so a P1 hit with missing
section can be retrieved but cannot be cited. Numerical checks are conservative,
bounded local checks plus the required verifier; they are not a universal proof
of semantic entailment. First-user-token timing is recorded only on actual
validated-generator delivery, distinct from internal TTFT and handle completion.
No P5 control plane, answer cache, /ask API, loadgen or real Groq requests belong
to this task. All configurable starting thresholds are assumed/chosen, uncalibrated.
