import pytest

from relay.app.delivery import iter_validated_answer
from relay.app.models import State
from relay.app.orchestrator import Orchestrator
from relay.app.settings import pipeline_settings
from relay.app.state import InvalidTransition, StateMachine
from relay.tools.executor import ToolExecutor
from tests.fixtures.app_provider import ScriptedProvider, verdicts
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock
from tests.test_app_validation import example


async def setup(responses: list) -> tuple[Orchestrator, ScriptedProvider]:
    c = clock()
    provider = ScriptedProvider(responses)
    executor = ToolExecutor(fixture_engine().search_papers, clock=c)
    return Orchestrator(provider, executor, model="FAKE", clock=c), provider


async def test_complete_answer_only_after_verification() -> None:
    draft, _ = await example()
    app, provider = await setup([draft.model_dump_json(), verdicts("supported")])
    outcome = await app.handle("alpha", setting="A", request_id="r")
    assert outcome.state == State.RELEASED and outcome.status == "ok"
    assert outcome.draft_calls == outcome.verifier_calls == 1 and outcome.retries == 0
    assert outcome.tokens_in == 20 and outcome.tokens_out == 10 and outcome.counted_tokens == 26
    assert outcome.first_token_ms is None
    assert len(provider.requests) == 2
    assert (
        "".join([p async for p in iter_validated_answer(outcome, clock=app.clock)]) == outcome.text
    )
    assert outcome.first_user_token_at is not None
    assert outcome.states == [
        State.ARRIVED,
        State.INTERPRETED,
        State.RETRIEVED,
        State.GATE,
        State.DRAFTING,
        State.VALIDATING,
        State.RELEASED,
    ]


async def test_one_repair_then_abstention_never_releases_rejected_text() -> None:
    draft, _ = await example()
    app, provider = await setup(
        [
            draft.model_dump_json(),
            verdicts("unsupported"),
            draft.model_dump_json(),
            verdicts("unsupported"),
        ]
    )
    outcome = await app.handle("alpha", setting="A")
    assert outcome.state == State.ABSTAINED and outcome.status == "error"
    assert outcome.repair_calls == 1 and outcome.verifier_calls == 2
    assert outcome.provider_calls == 4 and outcome.tokens_in == 40
    assert "FAKE alpha evidence" not in outcome.text
    assert provider.requests[0].messages[0] == provider.requests[2].messages[0]


async def test_empty_evidence_refuses_locally_and_weak_can_still_draft() -> None:
    app, provider = await setup([])
    outcome = await app.handle("zqxvbn", setting="A")
    assert outcome.state == State.REFUSED_LOCAL and outcome.status == "ok"
    assert outcome.validated_refusal and outcome.tokens_in == 0 and not provider.requests
    draft, _ = await example()
    app, provider = await setup([draft.model_dump_json(), verdicts("supported")])
    app.config = pipeline_settings().model_copy(update={"gate_min_keyword_coverage": 1})
    outcome = await app.handle("alpha unfamiliar", setting="A")
    assert outcome.state == State.RELEASED and len(provider.requests) == 2
    assert outcome.states.count(State.BROADEN) <= 2


def test_invalid_and_terminal_transitions_fail_explicitly() -> None:
    machine = StateMachine()
    with pytest.raises(InvalidTransition):
        machine.move(State.RELEASED)
    machine.move(State.FAILED)
    with pytest.raises(InvalidTransition):
        machine.move(State.DRAFTING)
