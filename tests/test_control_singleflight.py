import asyncio

import pytest

from relay.app.singleflight import SingleFlight
from relay.control.settings import load_policy
from tests.fixtures.app_provider import verdicts
from tests.standin_helpers import clock
from tests.test_app_orchestrator import setup
from tests.test_app_validation import example


async def test_twenty_identical_waiters_share_one_computation_and_owner_can_cancel() -> None:
    draft, _ = await example()
    app, provider = await setup([draft.model_dump_json(), verdicts("supported")])
    gate = asyncio.Event()
    flight = SingleFlight(load_policy().cache, clock())
    computations = 0

    async def compute() -> object:
        nonlocal computations
        computations += 1
        await gate.wait()
        return await app.handle("alpha", setting="A")

    tasks = [asyncio.create_task(flight.run("same", compute, 10**12)) for _ in range(20)]
    await asyncio.sleep(0)
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    gate.set()
    results = await asyncio.gather(*tasks[1:])
    assert computations == 1 and len(provider.requests) == 2
    assert all(joined for _, joined in results)
    results[0][0].citations[0].title = "FAKE mutation"
    assert results[1][0].citations[0].title != "FAKE mutation"
    assert (await flight.snapshot())["waiters"] == 0


async def test_last_waiter_cancellation_owns_computation_cleanup() -> None:
    flight = SingleFlight(load_policy().cache, clock())
    started, closed = asyncio.Event(), asyncio.Event()

    async def compute() -> object:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    task = asyncio.create_task(flight.run("same", compute, 10**12))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and (await flight.snapshot())["operations"] == 0
