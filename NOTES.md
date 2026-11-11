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
