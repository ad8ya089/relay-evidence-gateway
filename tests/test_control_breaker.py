import asyncio

import pytest

from relay.control.breaker import CircuitBreaker
from relay.control.settings import load_policy
from tests.standin_helpers import clock


@pytest.mark.parametrize("probe_failure", [None, "server"])
async def test_open_half_open_single_probe_and_recovery(probe_failure: str | None) -> None:
    c = clock()
    breaker = CircuitBreaker(load_policy().breaker, c)
    for _ in range(5):
        await breaker.record("a", "owner", "server")
    assert (await breaker.snapshot("a")).state == "open"
    assert not await breaker.allow("a", "x") and await breaker.allow("b", "x")
    c.advance_ms(4999)
    assert not await breaker.allow("a", "x")
    c.advance_ms(1)
    results = await asyncio.gather(*(breaker.allow("a", str(i)) for i in range(20)))
    assert sum(results) == 1
    await breaker.record("a", "0", probe_failure)
    assert (await breaker.snapshot("a")).state == ("closed" if probe_failure is None else "open")


@pytest.mark.parametrize(
    "kind",
    [
        "auth",
        "bad_request",
        "rate_limit_day",
        "rate_limit_unknown",
        "rate_limit_minute",
        "invalid_response",
    ],
)
async def test_non_outages_do_not_open_breaker(kind: str) -> None:
    breaker = CircuitBreaker(load_policy().breaker, clock())
    for _ in range(10):
        await breaker.record("a", "r", kind)
    assert (await breaker.snapshot("a")).state == "closed"


async def test_consecutive_failures_must_fit_the_window() -> None:
    c = clock()
    breaker = CircuitBreaker(load_policy().breaker, c)
    for _ in range(4):
        await breaker.record("a", "r", "timeout")
    c.advance_ms(20001)
    await breaker.record("a", "r", "timeout")
    assert (await breaker.snapshot("a")).failures == 1
    await breaker.record("a", "r", None)
    assert (await breaker.snapshot("a")).failures == 0
