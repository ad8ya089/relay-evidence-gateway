PYTHON ?= uv run --locked python
EVAL_ARGS ?=
INGEST_ARGS ?= --dense
PROBE_ARGS ?= --dry-run

.PHONY: ingest retr-eval test check standin relay probe probe-shapes eval-real calibrate capacity scenario scenarios aggregate timeline
ingest:
	$(PYTHON) -m relay.retrieval.ingest $(INGEST_ARGS)
retr-eval:
	$(PYTHON) -m eval.retrieval_offline $(EVAL_ARGS)
probe:
	$(PYTHON) -m scripts.limit_probe $(PROBE_ARGS)
probe-shapes:
	$(PYTHON) -m scripts.probe_groq $(PROBE_ARGS)
test:
	$(PYTHON) -m pytest
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest
	@echo P1/P2 checks only: full submission checks arrive in P7.

# P3-P7 stubs: fail explicitly instead of implying success.
standin relay eval-real calibrate capacity scenario scenarios aggregate timeline:
	$(PYTHON) -m scripts.pending $@
