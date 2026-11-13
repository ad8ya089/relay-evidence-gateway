PYTHON ?= uv run --locked python
EVAL_ARGS ?=
INGEST_ARGS ?= --dense
PROBE_ARGS ?= --dry-run
PROFILE ?= development
STANDIN_ARGS ?=
CALIBRATE_ARGS ?=
QUESTION ?= Explain encoder-decoder attention
PIPELINE_ARGS ?= --setting A

.PHONY: ingest retr-eval test check standin relay probe probe-shapes eval-real calibrate capacity scenario scenarios aggregate timeline pipeline-smoke
ingest:
	$(PYTHON) -m relay.retrieval.ingest $(INGEST_ARGS)
retr-eval:
	$(PYTHON) -m eval.retrieval_offline $(EVAL_ARGS)
probe:
	$(PYTHON) -m scripts.limit_probe $(PROBE_ARGS)
probe-shapes:
	$(PYTHON) -m scripts.probe_groq $(PROBE_ARGS)
standin:
	$(PYTHON) -m standin.service --profile $(PROFILE) $(STANDIN_ARGS)
calibrate:
	$(PYTHON) -m standin.calibrate $(CALIBRATE_ARGS)
pipeline-smoke:
	$(PYTHON) -m relay.app.smoke --question "$(QUESTION)" $(PIPELINE_ARGS)
test:
	$(PYTHON) -m pytest
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest
	@echo P1/P2/P3/P4 checks only: full submission checks arrive in P7.

# P4-P7 stubs: fail explicitly instead of implying success.
relay eval-real capacity scenario scenarios aggregate timeline:
	$(PYTHON) -m scripts.pending $@
