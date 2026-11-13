"""FAKE deterministic fixture controls around the actual P1/P2/P3 components."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from relay.app.orchestrator import Orchestrator
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.control.ledger import Ledger
from relay.obs.events import EventWriter
from relay.providers.groq import GroqCompatibleProvider
from relay.retrieval.search import RetrievalEngine
from relay.tools.executor import ToolExecutor
from standin.service import create_app
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock, profile


@asynccontextmanager
async def integrated(
    tmp_path: Path,
    *,
    reject: bool = False,
    engine: RetrievalEngine | None = None,
    config: PipelineSettings | None = None,
) -> AsyncIterator[tuple[Orchestrator, Ledger, Path]]:
    c = clock()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)
        await asyncio.sleep(0)

    p = profile(faults__verify_fail_prob=int(reject))
    app = create_app(profile=p, clock=c, sleep=sleep)
    events_path = tmp_path / "events.jsonl"
    events = EventWriter(events_path, clock=c)
    with Ledger(tmp_path / "usage.sqlite3", clock=c) as ledger:
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            p.model,
            clock=c,
            events=events,
            ledger=ledger,
            transport=httpx.ASGITransport(app=app),
        ) as provider:
            executor = ToolExecutor(
                (engine or fixture_engine()).search_papers,
                clock=c,
                events=events,
                config=config or pipeline_settings(),
            )
            yield (
                Orchestrator(
                    provider,
                    executor,
                    model=p.model,
                    clock=c,
                    events=events,
                    ledger=ledger,
                    config=config,
                    run_id="FAKE-p4-integration",
                ),
                ledger,
                events_path,
            )
