import json
from pathlib import Path

import pytest

from relay.app.calls import CallRunner, RequestCalls
from relay.app.delivery import DeliveryError, iter_validated_answer
from relay.app.models import Draft, State
from relay.app.orchestrator import Orchestrator
from relay.app.prompts import Prompts
from relay.app.settings import pipeline_settings
from relay.app.validation import validate_evidence
from relay.app.verify import verify
from relay.obs.events import Event, EventContext, EventWriteError, EventWriter
from relay.retrieval.models import RetrievalResult
from relay.tools.executor import ToolExecutor
from tests.fixtures.app_provider import ScriptedProvider, verdicts
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock
from tests.test_app_validation import example


async def test_tool_retry_is_shared_across_all_broadening_queries() -> None:
    c = clock()
    _, hits = await example()
    calls = 0

    async def search(query: str, *args: object, **kwargs: object) -> RetrievalResult:
        nonlocal calls
        calls += 1
        if calls != 2:
            raise RuntimeError("FAKE transient tool failure")
        return RetrievalResult(status="ok", hits=hits, setting="A", query_used=query, method="bm25")

    config = pipeline_settings().model_copy(update={"gate_min_keyword_coverage": 1})
    provider = ScriptedProvider([])
    app = Orchestrator(
        provider,
        ToolExecutor(search, clock=c, config=config),
        model="FAKE",
        clock=c,
        config=config,
    )
    outcome = await app.handle("alpha unfamiliar attention", setting="A")
    assert calls == 3 and outcome.tool_retries == 1
    assert outcome.state == State.FAILED and outcome.provider_calls == outcome.retries == 0


async def test_unexpected_provider_exception_is_terminal_without_repair() -> None:
    c = clock()
    provider = ScriptedProvider([RuntimeError("FAKE unexpected provider exception")])
    app = Orchestrator(
        provider, ToolExecutor(fixture_engine().search_papers, clock=c), model="FAKE", clock=c
    )
    outcome = await app.handle("alpha", setting="A")
    assert outcome.state == State.FAILED and not outcome.releasable
    assert outcome.provider_calls == 1 and outcome.repair_calls == 0
    assert not outcome.usage.complete and outcome.tokens_in is None


async def test_positive_but_insufficient_repair_budget_abstains_without_extra_call() -> None:
    c = clock()
    draft, _ = await example()
    provider = ScriptedProvider([draft.model_dump_json(), verdicts("unsupported")])
    app = Orchestrator(
        provider, ToolExecutor(fixture_engine().search_papers, clock=c), model="FAKE", clock=c
    )
    outcome = await app.handle("alpha", setting="A", deadline_ms=1000)
    assert outcome.state == State.ABSTAINED and outcome.provider_calls == 2
    assert outcome.failure_reason == "repair_deadline_insufficient" and outcome.repair_calls == 0


async def test_approved_answer_expires_before_delivery_without_first_user_token() -> None:
    c = clock()
    draft, _ = await example()
    provider = ScriptedProvider([draft.model_dump_json(), verdicts("supported")])
    app = Orchestrator(
        provider, ToolExecutor(fixture_engine().search_papers, clock=c), model="FAKE", clock=c
    )
    outcome = await app.handle("alpha", setting="A", deadline_ms=1000)
    assert outcome.state == State.RELEASED and outcome.releasable
    c.advance_ms(1000)
    with pytest.raises(DeliveryError):
        _ = [part async for part in iter_validated_answer(outcome, clock=c)]
    assert outcome.first_user_token_at is None


@pytest.mark.parametrize("indices", [[0, 0], [0], [0, 1, 2], [1, 2]])
async def test_missing_duplicate_and_extra_verdict_indices_never_pass(indices: list[int]) -> None:
    c = clock()
    draft, hits = await example()
    draft.claims.append(draft.claims[0].model_copy(deep=True))
    response = json.dumps(
        {"verdicts": [{"i": i, "v": "supported", "why": "FAKE"} for i in indices]}
    )
    runner = CallRunner(ScriptedProvider([response]), "FAKE", clock=c, config=pipeline_settings())
    result = await verify(
        draft,
        validate_evidence(draft, hits),
        runner,
        RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12),
        Prompts(),
    )
    assert result.status == "verify_invalid" and not result.verifier_passed


async def test_model_refusal_is_canonical_and_does_not_call_verifier() -> None:
    c = clock()
    response = Draft(
        status="insufficient_evidence", claims=[], note="FAKE internal note"
    ).model_dump_json()
    provider = ScriptedProvider([response])
    app = Orchestrator(
        provider, ToolExecutor(fixture_engine().search_papers, clock=c), model="FAKE", clock=c
    )
    outcome = await app.handle("alpha", setting="A")
    assert (
        outcome.state == State.REFUSED_MODEL
        and outcome.status == "ok"
        and outcome.validated_refusal
    )
    assert outcome.provider_calls == 1 and outcome.verifier_calls == 0
    assert "internal note" not in outcome.text


@pytest.mark.parametrize("where", ["arrived", "release_ready"])
async def test_event_persistence_failure_cannot_authorize_output(
    tmp_path: Path, where: str
) -> None:
    c = clock()
    draft, _ = await example()

    class BrokenEvents(EventWriter):
        def write(self, context: EventContext, ev: str, **fields: object) -> Event:
            if ev == where:
                raise EventWriteError("FAKE persistence failure")
            return super().write(context, ev, **fields)

    events = BrokenEvents(tmp_path / "events", clock=c)
    provider = ScriptedProvider([draft.model_dump_json(), verdicts("supported")])
    app = Orchestrator(
        provider,
        ToolExecutor(fixture_engine().search_papers, clock=c),
        model="FAKE",
        clock=c,
        events=events,
    )
    outcome = await app.handle("alpha", setting="A")
    assert outcome.state == State.FAILED and outcome.failure_reason == "event_persistence"
    assert not outcome.releasable and not outcome.text and not outcome.citations
    assert outcome.states.count(State.FAILED) == 1
    with pytest.raises(DeliveryError):
        _ = [part async for part in iter_validated_answer(outcome, clock=c)]


@pytest.mark.parametrize("limit", [1, 3])
async def test_invalid_request_and_configured_maximum_stop_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    c = clock()
    provider = ScriptedProvider([])
    config = pipeline_settings().model_copy(update={"max_question_chars": limit})
    monkeypatch.setattr("relay.app.models.pipeline_settings", lambda: config)
    app = Orchestrator(
        provider,
        ToolExecutor(fixture_engine().search_papers, clock=c),
        model="FAKE",
        clock=c,
        config=config,
    )
    for question in ("", "alpha"):
        outcome = await app.handle(question, setting="A")
        assert outcome.state == State.FAILED and outcome.failure_reason == "invalid_request"
    assert not provider.requests


async def test_numeric_claim_cannot_smuggle_an_unpinned_version() -> None:
    draft, hits = await example()
    draft.claims[0].text = "The Transformer v999 establishes this claim."
    result = validate_evidence(draft, hits)
    assert not result.deterministic_passed
