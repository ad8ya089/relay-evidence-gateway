# 03. Retrieval and evaluation

## 1. Corpus (pin exactly; no "latest")

| Key | Paper | arXiv | Version | PDF URL |
|-----|-------|-------|---------|---------|
| `transformer` | Attention Is All You Need | 1706.03762 | **v7** | https://arxiv.org/pdf/1706.03762v7 |
| `moshi` | Moshi: a speech-text foundation model for real-time dialogue | 2410.00037 | **v1** | https://arxiv.org/pdf/2410.00037v1 |
| `dynamic_chunking` | Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 | **v2** | https://arxiv.org/pdf/2507.07955v2 |

`ingest.py` must: download, save to `data/papers/`, compute sha256, **assert the arXiv stamp on page 1 contains the expected `arXiv:<id>v<n>`** (fail loudly if not), and write `corpus_manifest.json` (id, version, url, sha256, page count, fetch date). Every chunk and every citation carries `arxiv_id` + `arxiv_version`. Compute `corpus_hash` = sha256 of the three file hashes; it goes into the cache key and the run manifest.

## 2. Ingestion and chunking

1. PyMuPDF `page.get_text("blocks")`, sorted in reading order (two-column papers: sort by column then y; verify by eye on page 1 of each paper).
2. Detect headings with a regex for numbered sections (`^\d+(\.\d+)*\s+[A-Z]`) plus a font-size/bold heuristic; keep a running `section_path` ("3.2.3 Applications of Attention in our Model"). References and appendix tables of hyperparameters can be chunked but flagged `kind: references|appendix`.
3. Chunk by paragraph within a section, packing to the target token size with overlap (count with `tiktoken` `o200k_base` as an approximation; say so).
4. Tables, equations, figure captions: keep the caption/table block as its own chunk flagged `kind: table|caption`, and also leave its text in the surrounding chunk if short. **Manually inspect** extraction for: Transformer Table 1 (complexity per layer) and §3.2.3; Moshi latency passage; Dynamic Chunking's main method section. Record what broke in your notes; it feeds "how this library will mislead a naive retriever".
5. Chunk ID: `{paper}-v{ver}-p{page}-c{idx}`. Store `text`, `section`, `page_start`, `page_end`, `kind`, `n_tokens`.
6. Page numbers = **PDF page index (1-based)**; say so in the write-up.

## 3. Retrieval settings

| Setting | Piece size / overlap | Fetch | Method | Re-rank |
|---------|---------------------|-------|--------|---------|
| **A** small | 256 / 40 | top 6 | BM25 | no |
| **B** large | 512 / 64 | top 4 | BM25 | no |
| **C** hybrid + rerank | 384 / 48 | 12 candidates, **final 4** | BM25 + dense (bge-small-en-v1.5 via fastembed) fused by RRF | cross-encoder (fastembed) |
| D (offline only) | 384 / 48 | top 4 | BM25 + dense RRF | no |

C's final count is 4, not 5, to cap tokens. D exists only in the offline diagnostic so the rerank effect is isolated from the chunk-size effect without spending tokens. If the cross-encoder will not install in time: fall back to RRF only and report that honestly as "C without rerank".

Query handling: `paper_hint` from `interpret.py` is a **boost** (x1.3 score on that paper's chunks), not a hard filter, unless the question names a paper explicitly. A hard filter on a wrong guess is a silent retrieval failure.

Numeric/qualifier handling: if `numeric=true`, also add the neighbouring chunk (prev/next within the same section) of the top hit; this is cheap and targets the "number without its qualifier" failure.

Pressure mode shrinks fetch: A 6->3, B 4->3 (smaller output cap too), C 4->3.

## 4. Offline retrieval diagnostics (free, no tokens)

For each answerable question, `gold.jsonl` lists `gold_phrases` (1-3 short strings that must appear in the right passage, copied from the PDF by you) and the gold `paper`, `section`, `page`. Metrics per setting A/B/C/D:

- **hit@k**: some retrieved chunk contains all (or any, say which) gold phrases
- **MRR** of the first gold-containing chunk
- **wrong-paper@1**: top hit from a different paper than gold
- mean context tokens sent to the model

This table explains *why* a generation grade is wrong (retrieval miss vs model error vs validation) and costs nothing. Report it in a footnote or one extra line in 5a if space allows.

## 5. The 18-question evaluation set (YOU write it)

The brief says "questions you wrote yourself". Do not have an AI write or grade-key them. An AI may help you *check* your key against the PDF, but the gold claims must be ones you read in the paper.

Composition: **12 answerable** (4 per paper) + **6 unanswerable** (not supported by the three papers). That gives Table 3's "correctly refused %" a real denominator.

Format `eval/questions.jsonl` (one per line): `{"qid":"q07","question":"...","paper":"moshi","answerable":true,"type":"number|explain|compare|limitation|absent"}`.
Format `eval/gold.jsonl`: `{"qid":"q07","gold_claims":["..."],"gold_phrases":["..."],"paper":"moshi","version":1,"section":"...","page":7,"must_refuse":false,"notes":"..."}`.

Design each question for a purpose. Suggested mix (these are *prompts for your own thinking*, not finished questions; write your own wording and verify each against the PDF):

| Paper | Ideas to probe |
|-------|----------------|
| Transformer v7 | Which of the three uses of attention connects decoder to encoder, and where do keys and values come from? What exactly is masked in decoder self-attention and why? Reading the per-layer complexity comparison: when is self-attention cheaper than recurrence (it depends on a condition: check the paper)? A hyperparameter from the base model table. |
| Moshi v1 | What are the parallel streams? What does the "inner monologue" add? Distinguish the theoretical latency from the practical figure (look up both numbers and their conditions). A component of the codec or the temporal/depth transformer split. |
| Dynamic Chunking v2 | What is learned dynamically, and at what level of the model? How does the hierarchy compress and then restore the sequence? How does this differ from fixed tokenisation in the paper's own framing? An architecture detail or result you can find. |
| Unanswerable | Things that *sound* answerable: a recommended chunk size for PDF RAG (the DC paper's "chunking" is not that); today's Groq free-tier TPM; whether Vaani runs Moshi in production; a result for a benchmark none of the papers report; a detail from a later arXiv version you can show is not in the pinned one; a comparison across papers none of them makes. |

At least 3 questions should be **adversarial to a lexical retriever** (word overlap with the wrong passage) and at least 2 should need a **qualifier** to be correct. Those become your three naive-retriever failures.

## 6. Grading rubric (manual, by you)

| Grade | Definition |
|-------|-----------|
| `correct` | Answer is factually right per your gold claims **and** every citation points to a passage that actually supports its claim, with the right version |
| `wrong` | Incorrect, unsupported, mis-cited, wrong version, or answered when it should have refused |
| `refused_correctly` | The papers do not support an answer and the system said so |
| `refused_wrongly` | The papers support an answer but the system refused or said "could not verify" |

Rules: a citation that exists but does not support the claim is `wrong`. An abstention caused by `validation_failed_after_repair` on an answerable question is `refused_wrongly` for grading even though CSV `status` is `error`. Record **which stage failed** for every non-correct row in `eval/grades.csv` (`retrieval_miss | model_misread | validator_false_reject | validator_miss | refusal_threshold | budget`). Re-grade 5 random rows after two hours and report agreement.

Table 3 definitions: Questions asked = 18 per setting; Correct and cited % = `correct / 18` (answerable + unanswerable pooled; also give per-class numbers in the text if space allows); Wrong % = `wrong / 18`; Correctly refused % = `refused_correctly / 6`. State the denominators. With n = 18 a single question is 5.6 points, so say that differences under ~10 points are within noise.

## 7. Candidate failure modes to *test* (not to assert)

Run your questions against setting A and write down what you actually observe. These are hypotheses to check; only write the ones you saw.

1. **Lexical bleed on "chunking"/"tokens":** a RAG chunk-size question ranks the Dynamic Chunking paper first although the paper does not address PDF chunk sizes. Right behaviour: refuse.
2. **"Attention" polysemy:** "how does attention connect encoder and decoder" returns self-attention or masked-attention passages (or Moshi/H-Net passages that merely use attention) instead of the §3.2.3 encoder-decoder passage.
3. **Number without its qualifier:** Moshi's 160 ms (theoretical) retrieved without the ~200 ms practical figure, or the Transformer complexity table's O(.) entries without the n-versus-d condition.
4. **PDF extraction damage:** tables and equations flattened to unreadable text, so the right chunk exists but cannot be matched or quoted.
5. **Version drift:** a detail present only in a later arXiv version. Only relevant if you can show a real difference between the pinned and latest versions.

Pick the three you can demonstrate with a logged retrieval result and a question id.
