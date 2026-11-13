import pytest

from relay.app.calls import CallRunner, RequestCalls
from relay.app.models import Draft
from relay.app.prompts import Prompts
from relay.app.settings import pipeline_settings
from relay.app.validation import validate_evidence
from relay.app.verify import verify
from relay.providers.base import Message, ProviderError, ProviderFailure, Usage
from tests.fixtures.app_provider import ScriptedProvider, verdicts
from tests.standin_helpers import clock
from tests.test_app_validation import example


@pytest.mark.parametrize(
    "reply,expected",
    [
        (verdicts("supported"), "pass"),
        (verdicts("unsupported"), "fail_repairable"),
        (verdicts("overstated"), "fail_repairable"),
        ("{}", "verify_invalid"),
        ("garbage", "verify_invalid"),
        (verdicts(), "verify_invalid"),
        (verdicts("supported", "supported"), "verify_invalid"),
    ],
)
async def test_verifier_requires_complete_valid_verdicts(reply: str, expected: str) -> None:
    c = clock()
    draft, hits = await example()
    provider = ScriptedProvider([reply])
    runner = CallRunner(provider, "FAKE", clock=c, config=pipeline_settings())
    book = RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12)
    result = await verify(draft, validate_evidence(draft, hits), runner, book, Prompts())
    assert result.status == expected and len(provider.requests) == 1
    assert result.verifier_passed == (expected == "pass")
    assert book.totals().counted_tokens == 13


async def test_no_verifier_before_deterministic_validation_and_no_provider_retries() -> None:
    c = clock()
    draft, hits = await example()
    draft.claims[0].evidence[0].quote = "forged"
    provider = ScriptedProvider([])
    runner = CallRunner(provider, "FAKE", clock=c, config=pipeline_settings())
    book = RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12)
    assert not (
        await verify(draft, validate_evidence(draft, hits), runner, book, Prompts())
    ).verifier_passed
    assert not provider.requests
    usage = Usage(tokens_in=20, tokens_out=3, tokens_cached=2, total_tokens=23)
    failing = ScriptedProvider(
        [ProviderError(ProviderFailure(kind="server", status_code=500, usage=usage))]
    )
    runner = CallRunner(failing, "FAKE", clock=c, config=pipeline_settings())
    result = await runner.invoke(book, "draft", Prompts().draft("Q", hits), Draft)
    assert result.failure.kind == "server" and len(failing.requests) == 1
    assert book.totals().counted_tokens == 21 and book.totals().tokens_out == 3


async def test_unknown_usage_is_not_zero_and_deadline_stops_dispatch() -> None:
    c = clock()
    provider = ScriptedProvider([ProviderError(ProviderFailure(kind="disconnect"))])
    runner = CallRunner(provider, "FAKE", clock=c, config=pipeline_settings())
    book = RequestCalls(request_id="r", run_id="unit", arrival_ns=0, deadline_ns=10**12)
    await runner.invoke(book, "draft", [Message(role="user", content="FAKE")], Draft)
    assert book.totals().tokens_in is None and not book.totals().complete
    book.deadline_ns = 0
    assert (await runner.invoke(book, "draft", [], Draft)).failure.kind == "timeout"
    assert len(provider.requests) == 1
