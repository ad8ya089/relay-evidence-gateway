import asyncio
from pathlib import Path

import pytest

from relay.control.ledger import CallRecord, Ledger
from relay.control.limiter import Limiter, Need
from relay.control.settings import Policy, load_policy
from relay.providers.base import Usage
from tests.standin_helpers import clock

MODEL = "openai/gpt-oss-20b"


def policy(**changes: object) -> Policy:
    return Policy.model_validate(load_policy().model_dump() | changes)


async def test_reservations_are_atomic_and_unused_release_is_idempotent(tmp_path: Path) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(
            policy(limits={MODEL: {"rpm": 10, "tpm": 100, "rpd": 20, "tpd": 1000}}), ledger, c
        )
        accepted = await asyncio.gather(
            *(limiter.reserve(str(i), Need(model=MODEL, tokens=30, requests=2)) for i in range(20))
        )
        assert sum(accepted) == 3
        assert (await limiter.remaining(MODEL)).tpm == 10
        await limiter.release("0")
        await limiter.release("0")
        assert (await limiter.remaining(MODEL)).tpm == 40


@pytest.mark.parametrize("known", [True, False])
async def test_windows_cached_tokens_unknown_and_duplicate_reconciliation(
    tmp_path: Path, known: bool
) -> None:
    c = clock()
    c.set_utc(c.utc_now().replace(hour=12))
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(
            policy(limits={MODEL: {"rpm": 2, "tpm": 100, "rpd": 20, "tpd": 1000}}), ledger, c
        )
        need = Need(model=MODEL, tokens=40, requests=1)
        assert await limiter.reserve("r", need)
        await limiter.start_call("r", "call", need)
        usage = (
            Usage(tokens_in=50, tokens_out=20, tokens_cached=10, total_tokens=70)
            if known
            else Usage()
        )
        ledger.record_call(
            CallRecord(
                id="call",
                model=MODEL,
                run_id="FAKE",
                request_id="r",
                kind="draft",
                status="ok",
                real=False,
                session_id="FAKE",
                reserved_tokens=40,
                **usage.model_dump(
                    include={"tokens_in", "tokens_out", "tokens_cached", "total_tokens"}
                ),
            )
        )
        await limiter.reconcile("call", usage)
        await limiter.reconcile("call", Usage(tokens_in=1, tokens_out=1, tokens_cached=0))
        await limiter.release("r")
        view = await limiter.remaining(MODEL)
        assert view.tpm == (40 if known else 60)
        assert view.tpd == (940 if known else 960)
        assert view.uncertain != known
        c.advance_ms(60000)
        assert (await limiter.remaining(MODEL)).tpm == 100
        assert (await limiter.remaining(MODEL)).tpd == view.tpd
        c.advance_ms(86400000)
        assert (await limiter.remaining(MODEL)).tpd == 1000


def test_invalid_policy_fails_startup() -> None:
    with pytest.raises(ValueError):
        policy(queue_capacity=-1)
    with pytest.raises(ValueError):
        policy(pressure=load_policy().normal.model_dump())
