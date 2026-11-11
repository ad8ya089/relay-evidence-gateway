PYTHON ?= uv run --locked python
EVAL_ARGS ?=
INGEST_ARGS ?= --dense

.PHONY: ingest retr-eval test check standin relay probe eval-real calibrate capacity scenario scenarios aggregate timeline
ingest:
	$(PYTHON) -m relay.retrieval.ingest $(INGEST_ARGS)
retr-eval:
	$(PYTHON) -m eval.retrieval_offline $(EVAL_ARGS)
test:
	$(PYTHON) -m pytest
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest
	@echo P1 checks only: full submission checks arrive in P7.

# P2-P7 stubs: fail explicitly instead of implying success.
standin relay probe eval-real calibrate capacity scenario scenarios aggregate timeline:
	$(PYTHON) -m scripts.pending $@
