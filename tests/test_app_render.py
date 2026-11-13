import pytest

from relay.app.calls import CallRunner, RequestCalls
from relay.app.delivery import DeliveryError, _authorize, iter_validated_answer
from relay.app.models import Outcome, State
from relay.app.prompts import Prompts
from relay.app.render import RenderError, render_answer
from relay.app.settings import pipeline_settings
from relay.app.validation import validate_evidence
from relay.app.verify import verify
from tests.fixtures.app_provider import ScriptedProvider, verdicts
from tests.standin_helpers import clock
from tests.test_app_validation import example


async def test_citations_are_source_owned_and_validation_is_bound_to_draft() -> None:
    c = clock()
    draft, hits = await example()
    deterministic = validate_evidence(draft, hits)
    with pytest.raises(RenderError):
        render_answer(draft, hits, deterministic)
    runner = CallRunner(
        ScriptedProvider([verdicts("supported")]), "FAKE", clock=c, config=pipeline_settings()
    )
    validated = await verify(
        draft,
        deterministic,
        runner,
        RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12),
        Prompts(),
    )
    rendered = render_answer(draft, hits, validated)
    assert rendered.citations[0].arxiv_version == 7
    assert rendered.citations[0].arxiv_id == hits[0].arxiv_id
    assert rendered.citations[0].title == hits[0].title
    assert "v7" in rendered.text and "p.1" in rendered.text
    outcome = Outcome(
        request_id="r",
        run_id="unit",
        state=State.RELEASED,
        status="ok",
        text=rendered.text,
        citations=rendered.citations,
        validation=validated,
        releasable=True,
        setting="A",
        arrival_ns=0,
        deadline_ns=10**12,
    )
    with pytest.raises(DeliveryError):
        _ = [part async for part in iter_validated_answer(outcome, clock=c)]
    _authorize(outcome)
    assert outcome.first_user_token_at is None
    assert (
        "".join([part async for part in iter_validated_answer(outcome, clock=c)]) == rendered.text
    )
    assert outcome.first_user_token_at == 0 and outcome.first_token_ms == 0
    outcome.text = "Unvalidated modification"
    with pytest.raises(DeliveryError):
        _ = [part async for part in iter_validated_answer(outcome, clock=c)]
    draft.claims[0].text = "A changed unverified claim"
    with pytest.raises(RenderError):
        render_answer(draft, hits, validated)


async def test_no_delivery_before_validation_or_after_deadline() -> None:
    outcome = Outcome(
        request_id="r",
        run_id="unit",
        state=State.DRAFTING,
        status="error",
        text="SECRET DRAFT",
        setting="A",
        arrival_ns=0,
        deadline_ns=10**12,
    )
    with pytest.raises(DeliveryError):
        _authorize(outcome)
    with pytest.raises(DeliveryError):
        _ = [part async for part in iter_validated_answer(outcome, clock=clock())]


async def test_shared_citation_is_deduplicated_without_losing_claim_associations() -> None:
    draft, hits = await example()
    draft.claims.append(draft.claims[0].model_copy(deep=True))
    runner = CallRunner(
        ScriptedProvider([verdicts("supported", "supported")]),
        "FAKE",
        clock=clock(),
        config=pipeline_settings(),
    )
    validated = await verify(
        draft,
        validate_evidence(draft, hits),
        runner,
        RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12),
        Prompts(),
    )
    rendered = render_answer(draft, hits, validated)
    identifier = draft.claims[0].evidence[0].chunk_id
    assert len(rendered.citations) == 1
    assert rendered.claim_citations == {0: [identifier], 1: [identifier]}
    assert rendered.text.count(rendered.citations[0].display) == 2
