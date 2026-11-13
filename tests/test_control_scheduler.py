import asyncio

import pytest

from relay.control.scheduler import ProviderScheduler, SchedulerRejected
from relay.control.settings import load_policy
from tests.standin_helpers import clock


async def test_verifier_repair_priority_and_draft_starvation_protection() -> None:
    c = clock()
    config = load_policy().scheduler.model_copy(update={"max_priority_burst": 2})
    scheduler = ProviderScheduler(1, config, c)
    order = []

    async def work(kind: str) -> None:
        async with scheduler.slot(kind, 10**12):
            order.append(kind)
            await asyncio.sleep(0)

    async with scheduler.slot("draft", 10**12):
        tasks = [
            asyncio.create_task(work(kind)) for kind in ["draft", "repair", "verify", "verify"]
        ]
        await asyncio.sleep(0)
        assert (await scheduler.snapshot()).waiters == 4
    await asyncio.gather(*tasks)
    assert order == ["verify", "verify", "draft", "repair"]
    assert (await scheduler.snapshot()).inflight == 0


async def test_cancelled_and_expired_waiters_release_slots_and_queue_bounds() -> None:
    c = clock()
    config = load_policy().scheduler.model_copy(update={"max_waiters": 1})
    scheduler = ProviderScheduler(1, config, c)

    async def work() -> None:
        async with scheduler.slot("verify", 10**12):
            raise AssertionError("Cancelled waiter must not execute")

    async with scheduler.slot("draft", 10**12):
        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        with pytest.raises(SchedulerRejected):
            async with scheduler.slot("draft", 10**12):
                pass
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await scheduler.snapshot()).waiters == 0
    assert (await scheduler.snapshot()).inflight == 0
    with pytest.raises(SchedulerRejected):
        async with scheduler.slot("draft", 0):
            pass
