import asyncio
from pathlib import Path

import pytest

from relay.app.models import Outcome, Question, State
from relay.control.admission import Reject
from relay.providers.base import HeaderSnapshot
from tests.fixtures.control import controlled
from tests.test_control_limiter import policy


@pytest.mark.parametrize("pressure", [False, True])
async def test_modes_preserve_verification_and_bound_repairs(
    tmp_path: Path, pressure: bool
) -> None:
    async with controlled(tmp_path, reject=True) as plane:
        if pressure:
            await plane.limiter.update_from_headers(
                plane.model, HeaderSnapshot(observed_ns=0, remaining_tokens_tpm=30000)
            )
        seen = []
        original = plane.factory

        def factory(config: object, execution: object) -> object:
            seen.append(config)
            return original(config, execution)

        plane.factory = factory
        outcome = await plane.ask(Question(text="alpha", setting="A"))
        assert isinstance(outcome, Outcome) and outcome.state == State.ABSTAINED
        assert outcome.repair_calls == (0 if pressure else 1)
        assert outcome.verifier_calls == (1 if pressure else 2)
        if pressure:
            assert len(outcome.supplied_chunk_ids) <= 3
        assert seen[0].evidence_limit == (3 if pressure else None)
        assert seen[0].draft_max_completion_tokens == (280 if pressure else 450)
        assert not plane.limiter.reservations
        usage = plane.ledger.daily_usage(plane.model, real=False)
        assert usage.known_counted_tokens == outcome.counted_tokens


async def test_cache_precedes_exhaustion_and_no_cache_bypasses_coalescing(tmp_path: Path) -> None:
    cache = policy().cache.model_copy(update={"enabled": True}).model_dump()
    async with controlled(tmp_path, changes={"cache": cache}) as plane:
        first = await plane.ask(Question(text="alpha", setting="A", request_id="owner"))
        assert first.state == State.RELEASED
        plane.limiter._limits[plane.model] = plane.limiter._limits[plane.model].model_copy(
            update={"tpd": 1}
        )
        cached = await plane.ask(Question(text="alpha", setting="A", request_id="cached"))
        assert cached.state == State.CACHE_HIT and cached.provider_calls == 0
        rejected = await plane.ask(Question(text="alpha", setting="A"), cache_enabled=False)
        assert isinstance(rejected, Reject) and rejected.reason == "daily_budget_exhausted"
        assert plane.ledger.daily_usage(plane.model, real=False).attempts == 2


async def test_twenty_identical_questions_use_one_actual_p3_pipeline(tmp_path: Path) -> None:
    cache = policy().cache.model_copy(update={"enabled": True}).model_dump()
    async with controlled(tmp_path, changes={"cache": cache}) as plane:
        outcomes = await asyncio.gather(
            *(plane.ask(Question(text="alpha", setting="A", request_id=str(i))) for i in range(20))
        )
        assert all(isinstance(o, Outcome) and o.status == "ok" for o in outcomes)
        assert len({o.request_id for o in outcomes}) == 20
        assert plane.ledger.daily_usage(plane.model, real=False).attempts == 2
        assert sum(o.provider_calls for o in outcomes) == 2
        assert (await plane.cache.snapshot())["writes"] == 1
        assert not plane.limiter.reservations
