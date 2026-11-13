import asyncio
import random
from pathlib import Path

import pytest

from relay.app.settings import pipeline_settings
from relay.obs.events import EventContext, EventWriter
from relay.retrieval.models import RetrievalResult
from relay.tools.executor import ToolExecutor
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock


async def test_real_p1_interface_and_empty(tmp_path: Path) -> None:
    c = clock()
    executor = ToolExecutor(
        fixture_engine().search_papers, clock=c, events=EventWriter(tmp_path / "events", clock=c)
    )
    context = EventContext(rid="r", run="unit", arrival_ns=0)
    result = await executor.execute("alpha", setting="A", deadline_ns=10**12, context=context)
    assert result.status == "ok" and result.hits[0].arxiv_version == 7
    empty = await executor.execute("zqxvbn", setting="A", deadline_ns=10**12, context=context)
    assert empty.status == "empty" and not empty.hits


async def test_one_transient_retry_and_seeded_fault_injection() -> None:
    c, calls = clock(), []

    async def search(query: str, *args: object, **kwargs: object) -> RetrievalResult:
        calls.append(query)
        return RetrievalResult(status="empty", setting="A", method="bm25", query_used=query)

    config = pipeline_settings().model_copy(update={"tool_fault_prob": 0.5})
    executor = ToolExecutor(search, clock=c, rng=random.Random(1), config=config)
    result = await executor.execute(
        "alpha",
        setting="A",
        deadline_ns=10**12,
        context=EventContext(rid="r", run="unit", arrival_ns=0),
    )
    assert result.status == "empty" and result.attempt == 2 and calls == ["alpha"]


@pytest.mark.parametrize("mode", ["malformed", "timeout", "error"])
async def test_failures_are_not_empty_success(mode: str) -> None:
    c, calls = clock(), []

    async def search(query: str, *args: object, **kwargs: object) -> object:
        calls.append(query)
        if mode == "malformed":
            return {"hits": []}
        if mode == "timeout":
            await asyncio.sleep(1)
        raise RuntimeError("FAKE backend failure")

    config = pipeline_settings().model_copy(update={"tool_timeout_ms": 10})
    result = await ToolExecutor(search, clock=c, config=config).execute(
        "alpha",
        setting="A",
        deadline_ns=10**12,
        context=EventContext(rid="r", run="unit", arrival_ns=0),
    )
    assert result.status == ("timeout" if mode == "timeout" else "error")
    assert len(calls) == (2 if mode == "error" else 1)


async def test_deadline_stops_tool_retry() -> None:
    c, calls = clock(), []

    async def search(query: str, *args: object, **kwargs: object) -> RetrievalResult:
        calls.append(query)
        c.advance_ms(50)
        return RetrievalResult(status="error", setting="A", method="bm25", query_used=query)

    result = await ToolExecutor(search, clock=c).execute(
        "q",
        setting="A",
        deadline_ns=40_000_000,
        context=EventContext(rid="r", run="unit", arrival_ns=0),
    )
    assert result.status == "timeout" and calls == ["q"]
