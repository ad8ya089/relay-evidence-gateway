"""Actual P2/P3/P4 components with deterministic, explicitly synthetic controls."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from relay.app.cache import AnswerCache
from relay.app.singleflight import SingleFlight
from relay.control.admission import Admission
from relay.control.breaker import CircuitBreaker
from relay.control.ledger import Ledger
from relay.control.limiter import Limiter
from relay.control.plane import ControlPlane
from relay.control.retry import RetryBudget
from relay.control.scheduler import ProviderScheduler
from relay.obs.events import EventWriter
from relay.providers.groq import GroqCompatibleProvider
from relay.tools.executor import ToolExecutor
from standin.service import create_app
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock, profile
from tests.test_control_limiter import policy


@asynccontextmanager
async def controlled(
    tmp_path: Path, *, changes: dict[str, object] | None = None, reject: bool = False
) -> AsyncIterator[ControlPlane]:
    c = clock()
    c.set_utc(c.utc_now().replace(hour=12))
    config = policy(**(changes or {}))
    p = profile(faults__verify_fail_prob=int(reject))

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)
        await asyncio.sleep(0)

    service = create_app(profile=p, clock=c, sleep=sleep)
    events = EventWriter(tmp_path / "events", clock=c)
    cache = AnswerCache(tmp_path / "cache", config.cache, c)
    flight = SingleFlight(config.cache, c)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            p.model,
            ledger=ledger,
            events=events,
            clock=c,
            transport=httpx.ASGITransport(app=service),
        ) as provider:
            limiter = Limiter(config, ledger, c)
            breaker = CircuitBreaker(config.breaker, c)
            plane = ControlPlane(
                policy=config,
                clock=c,
                provider=provider,
                ledger=ledger,
                events=events,
                executor=ToolExecutor(fixture_engine().search_papers, clock=c, events=events),
                model=p.model,
                corpus_hash="FAKE-corpus",
                cache=cache,
                singleflight=flight,
                limiter=limiter,
                admission=Admission(config, limiter, breaker, c),
                scheduler=ProviderScheduler(config.max_inflight_provider, config.scheduler, c),
                retry_budget=RetryBudget(config.retry),
                breaker=breaker,
                sleep=sleep,
            )
            try:
                yield plane
            finally:
                await flight.close()
                cache.close()
