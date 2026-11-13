# P7 reliability review

This is an agent review, not an independent human sign-off. No human grades were assigned.

Confirmed defects and regression evidence:

- A provider root URL was accepted by the new evaluator and produced HTTP 404 model calls.
  The initial 54 stand-in rows remain in raw data and CSV. The endpoint regression test
  failed with application creation occurring before validation, then passed after requiring
  the `/openai/v1` provider base path. A corrected, separate A/B/C run was executed.
- The new provider selector initially changed an existing rejection from ValueError to
  PermissionError. The unchanged P5 hardening test caught it in the full suite (483 passed,
  one failed). The original exception contract was restored without weakening origin checks.
- Secret scans must inspect extracted PDF text, not only compressed PDF bytes. A planted
  synthetic-key PDF regression now verifies detection. No actual key is in the fixture source.
- A header-only grades CSV received a duplicate header on its first human grade. A failing
  regression reproduced the extra row; checking file existence/size instead of row count fixed it.
- Submission reconciliation caught a missing provider-call count when converting the raw smoke
  outcome: Pydantic serialization omits its computed property. Conversion now counts dispatched
  raw call records, with a regression ensuring the real attempt is not represented as zero calls.

Checked boundaries:

- Stand-in stays default. Groq requires provider selection, environment permission and
  explicit CLI intent. Permission failures occur before credential lookup; no real fallback
  is allowed. A real-provider lifecycle test makes no network calls and checks durable guard proof.
- HTTP transport retries remain zero; retry ownership, semaphore priorities, quota locking,
  conservative header reconciliation and bounded queues are unchanged. Real smoke retries
  were disabled, concurrency one, and a cumulative estimated-token bound wrapped the durable guard.
- The one real logical smoke made one draft call. A quote failed deterministic substring
  validation. The normal repair budget rejected further work. No verifier ran, no citations
  were released, and only the sealed abstention was visible. This is not a quality success.
- Every actual attempt, including failed attempts, remains ledger-accounted. Absent cached
  usage remains unknown; it is not replaced with zero. The original connectivity row remains intact.
- Model runtime loads question text and source metadata only. Gold references live outside
  the corpus and are used only for dataset checks, offline reference alignment and human grading.
  All source associations were checked against actual pinned chunks and PDF hashes/version stamps.
- Percentiles exclude failures and use NumPy linear interpolation. Repeated evaluation runs
  remain separate in diagnostics; the latest corrected run is explicitly selected for Table 3.
  Missing grades are missing, not incorrect. A stand-in verifier is not evidence of real quality.
- Historical P6 raw data and the 84-row CSV are hash-anchored independently of new P7 outputs.
  The original 14 protected inputs remain unchanged. Package paths cannot escape their root.

Observed retrieval ranking traps (setting A):

- B01: `moshi-v1-p64-c325`, F.2 Misspelling, is a demo dialogue about a football player,
  ahead of the spoken-dialogue architecture passages. Conversational wording attracts example dialogue.
- B02: `moshi-v1-p62-c319`, F.1 General knowledge, is an entropy dialogue, ahead of Mimi's
  architecture passage. Information-related vocabulary does not establish semantic/acoustic mechanism support.
- C03: `dynamic_chunking-v2-p2-c7`, Introduction, is an overview rather than the requested
  smoothing/ratio-loss derivation. Broad conceptual overlap can outrank the mechanism.

These are top-one reference mismatches, not automatic answer grades or proof of a complete
top-k miss. A01 correctly retrieved the attention-applications section first and is not a failure.
All actual hits and rerank methods are in the offline retrieval observation artifact.

Unconfirmed or unmeasured:

- No real 429, recovery interval, concurrent Groq capacity or daily-reset behaviour was tested.
- No actual organization daily-remaining snapshot is supplied. Historical request/token headers
  do not establish current TPD remaining. A full A setting's conservative budget already exceeds
  the existing session cap, so real quality evaluation was blocked without consuming quota.
- Calibration has one draft TTFT fit, zero verifier samples and no complete paired latency
  distribution. The calibration module's broad `calibrated` status means some fields fitted;
  it does not justify calling the whole development stand-in calibrated.
- Official S1-S8 and a calibrated retry-worsening comparison remain unmeasured. Native regression
  tests demonstrate retry amplification mechanics, but are not substituted for official scenario results.
- Citation support still needs human review; literal source anchoring alone cannot prove entailment.
- Full conversation exports, Limits screenshot, Vaani feedback and human grades remain candidate inputs.
