import asyncio
import random

import pytest

from relay.app.calls import CallRunner, RequestCalls
from relay.app.models import Draft
from relay.app.settings import pipeline_settings
from relay.control.retry import NaiveRetryPolicy, RetryBudget, RetryController
from relay.control.settings import load_policy
from relay.providers.base import LLMRequest, Message, ProviderError, ProviderFailure, Usage
from tests.fixtures.app_provider import ScriptedProvider
from tests.standin_helpers import clock


async def retry_fixture(
    kind: str, retry_after: float | None = None
) -> tuple[RequestCalls, ScriptedProvider, list[float]]:
    c = clock()
    config = load_policy().retry
    budget = RetryBudget(config)
    for _ in range(10):
        await budget.original()
    delays = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)
        c.advance_ms(seconds * 1000)

    execution = RetryController(config, c, budget, rng=random.Random(7), sleep=sleep)
    provider = ScriptedProvider(
        [
            ProviderError(
                ProviderFailure(
                    kind=kind,
                    retry_after_ms=retry_after,
                    usage=Usage(tokens_in=20, tokens_out=3, tokens_cached=2, total_tokens=23),
                )
            ),
            "FAKE success",
        ]
    )
    runner = CallRunner(provider, "FAKE", clock=c, config=pipeline_settings(), execution=execution)
    book = RequestCalls(request_id="r", run_id="FAKE", arrival_ns=0, deadline_ns=8_000_000_000)
    await runner.invoke(book, "draft", [Message(role="user", content="FAKE")], Draft)
    return book, provider, delays


async def test_transient_failure_retries_once_with_reproducible_full_jitter_and_usage() -> None:
    first, provider, delays = await retry_fixture("server")
    second, _, again = await retry_fixture("server")
    assert len(provider.requests) == 2 and first.retries == second.retries == 1
    assert delays == again and 0 < delays[0] < 0.2
    assert first.totals().tokens_in == 30 and first.totals().tokens_out == 8
    assert provider.requests[0].metadata.call_id != provider.requests[1].metadata.call_id
    assert provider.requests[1].metadata.attempt == 2


@pytest.mark.parametrize(
    "kind", ["rate_limit_day", "rate_limit_unknown", "auth", "bad_request", "invalid_response"]
)
async def test_unrecoverable_failures_are_never_retried(kind: str) -> None:
    book, provider, delays = await retry_fixture(kind)
    assert book.retries == 0 and len(provider.requests) == 1 and not delays


async def test_retry_after_beyond_remaining_deadline_never_retries() -> None:
    book, provider, delays = await retry_fixture("rate_limit_minute", 7000)
    assert book.retries == 0 and len(provider.requests) == 1 and not delays


async def test_global_retry_budget_bounds_aggregate_amplification() -> None:
    c = clock()
    config = load_policy().retry
    budget = RetryBudget(config)
    execution = RetryController(config, c, budget, sleep=asyncio.sleep)
    failure = ProviderError(ProviderFailure(kind="server"))
    provider = ScriptedProvider([failure] * 30)
    runner = CallRunner(provider, "FAKE", clock=c, config=pipeline_settings(), execution=execution)
    books = []
    for i in range(20):
        book = RequestCalls(request_id=str(i), run_id="FAKE", arrival_ns=0, deadline_ns=10**12)
        await runner.invoke(book, "draft", [Message(role="user", content="FAKE")], Draft)
        books.append(book)
    assert sum(b.retries for b in books) == 2 and len(provider.requests) == 22
    assert (await budget.snapshot())["retries"] == 2


async def test_slow_success_is_not_duplicated_by_bounded_policy_but_naive_is_explicit() -> None:
    config = load_policy().retry.model_copy(update={"naive_timeout_ms": 5})
    active = peak = calls = 0

    async def slow(attempt: int) -> str:
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.04)
            return "FAKE"
        finally:
            active -= 1

    assert await NaiveRetryPolicy(config).run(slow) == "FAKE"
    assert peak > 1 and calls == 4 and active == 0
    active = peak = calls = 0

    class SlowProvider(ScriptedProvider):
        async def complete(self, request: LLMRequest) -> object:
            await slow(1)
            return await super().complete(request)

    c = clock()
    provider = SlowProvider(["FAKE"])
    runner = CallRunner(
        provider,
        "FAKE",
        clock=c,
        config=pipeline_settings(),
        execution=RetryController(config, c, RetryBudget(config)),
    )
    await runner.invoke(
        RequestCalls(request_id="r", run_id="FAKE", arrival_ns=0, deadline_ns=10**12),
        "draft",
        [Message(role="user", content="FAKE")],
        Draft,
    )
    assert peak == calls == 1 and active == 0
