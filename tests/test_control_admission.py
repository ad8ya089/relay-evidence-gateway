import asyncio
from pathlib import Path

import pytest

from relay.control.admission import Admission, Admit, Reject
from relay.control.breaker import CircuitBreaker
from relay.control.ledger import Ledger
from relay.control.limiter import Limiter
from tests.standin_helpers import clock
from tests.test_control_limiter import MODEL, policy


@pytest.mark.parametrize(
    "reason",
    [
        "queue_full",
        "projected_wait",
        "daily_budget_exhausted",
        "breaker_open",
        "deadline_infeasible",
    ],
)
async def test_predictable_rejections_reserve_nothing(tmp_path: Path, reason: str) -> None:
    c = clock()
    config = policy(max_active_requests=1, queue_capacity=0 if reason == "queue_full" else 2)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(config, ledger, c)
        breaker = CircuitBreaker(config.breaker, c)
        admission = Admission(config, limiter, breaker, c)
        if reason == "daily_budget_exhausted":
            limiter._limits[MODEL] = limiter._limits[MODEL].model_copy(update={"tpd": 1})
        if reason == "projected_wait":
            limiter._limits[MODEL] = limiter._limits[MODEL].model_copy(update={"tpm": 1})
        if reason == "breaker_open":
            for _ in range(5):
                await breaker.record(MODEL, "r", "server")
        if reason == "queue_full":
            assert isinstance(await admission.admit("first", "A", MODEL, 10**12), Admit)
        result = await admission.admit(
            "r", "A", MODEL, 1 if reason == "deadline_infeasible" else 10**12
        )
        assert isinstance(result, Reject) and result.reason == reason
        assert "r" not in limiter.reservations


async def test_queue_capacity_atomicity_and_cancellation_cleanup(tmp_path: Path) -> None:
    c = clock()
    config = policy(max_active_requests=1, queue_capacity=2, max_queue_wait_ms=6000)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(config, ledger, c)
        admission = Admission(config, limiter, CircuitBreaker(config.breaker, c), c)
        results = await asyncio.gather(
            *(admission.admit(str(i), "A", MODEL, 10**12) for i in range(20))
        )
        assert sum(isinstance(r, Admit) for r in results) == 3
        assert (await admission.snapshot())["peak_queue_length"] == 2
        queued = results[1]
        task = asyncio.create_task(admission.enter(queued))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await admission.release(queued.request_id)
        await admission.release("0")
        assert await admission.enter(results[2]) == 0
        await admission.release("2")
        assert not limiter.reservations and not admission.active and not admission.queue
