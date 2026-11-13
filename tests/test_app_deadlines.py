import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from relay.app.models import State
from relay.app.orchestrator import Orchestrator, PipelineCancelledError
from relay.app.render import RenderedAnswer, render_answer
from relay.control.ledger import Ledger
from relay.obs.clock import SystemClock
from relay.obs.events import Event, EventContext, EventWriter
from relay.providers.groq import GroqCompatibleProvider
from relay.retrieval.models import RetrievalResult
from relay.tools.executor import ToolExecutor
from standin.profile import Profile
from standin.service import create_app
from tests.fixtures.app_provider import ScriptedProvider, verdicts
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock, profile
from tests.standin_server import running, until_idle
from tests.test_app_validation import example


@pytest.mark.parametrize(
    "stage,expected_calls",
    [
        ("retrieval", 0),
        ("draft", 1),
        ("verification", 2),
        ("before_repair", 2),
        ("before_release", 2),
    ],
)
async def test_deadline_expires_at_every_boundary_without_more_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, expected_calls: int
) -> None:
    c = clock()
    draft, _ = await example()
    engine = fixture_engine()

    async def search(query: str, *args: object, **kwargs: object) -> RetrievalResult:
        if stage == "retrieval":
            c.advance_ms(100)
        return await engine.search_papers(query, *args, **kwargs)

    def late_draft(request: object) -> str:
        if stage == "draft":
            c.advance_ms(100)
        return draft.model_dump_json()

    def late_verification(request: object) -> str:
        if stage == "verification":
            c.advance_ms(100)
        return verdicts("unsupported" if stage == "before_repair" else "supported")

    class AdvancingEvents(EventWriter):
        def write(self, context: EventContext, ev: str, **fields: object) -> Event:
            result = super().write(context, ev, **fields)
            if stage == "before_repair" and ev == "validated":
                c.advance_ms(100)
            return result

    if stage == "before_release":

        def late_render(*args: object, **kwargs: object) -> RenderedAnswer:
            rendered = render_answer(*args, **kwargs)
            c.advance_ms(100)
            return rendered

        monkeypatch.setattr("relay.app.orchestrator.render_answer", late_render)
    provider = ScriptedProvider([late_draft, late_verification])
    events = AdvancingEvents(tmp_path / "events", clock=c)
    app = Orchestrator(
        provider, ToolExecutor(search, clock=c), model="FAKE", clock=c, events=events
    )
    outcome = await app.handle("alpha", setting="A", deadline_ms=50)
    assert outcome.state == State.TIMED_OUT and not outcome.releasable and not outcome.text
    assert len(provider.requests) == expected_calls
    assert outcome.repair_calls == 0 and outcome.first_user_token_at is None
    if expected_calls:
        assert outcome.tokens_in == expected_calls * 10
    if stage == "verification":
        assert outcome.stage_ms["verification"] >= 100


class HangingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered, self.closed = asyncio.Event(), asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"{"}}]}\n\n'
        self.entered.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.parametrize("external_cancel", [False, True])
async def test_timeout_and_cancellation_preserve_status_partial_timing_and_ledger(
    tmp_path: Path, external_cancel: bool
) -> None:
    stream = HangingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    p = profile()
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            p.model,
            transport=httpx.MockTransport(handler),
            ledger=ledger,
        ) as provider:
            app = Orchestrator(
                provider, ToolExecutor(fixture_engine().search_papers), model=p.model, ledger=ledger
            )
            task = asyncio.create_task(
                app.handle("alpha", setting="A", deadline_ms=5000 if external_cancel else 100)
            )
            await asyncio.wait_for(stream.entered.wait(), 2)
            if external_cancel:
                task.cancel()
                with pytest.raises(PipelineCancelledError) as cancelled:
                    await task
                outcome = cancelled.value.outcome
            else:
                outcome = await task
        assert outcome.state == State.TIMED_OUT and outcome.provider_calls == 1
        assert stream.closed.is_set()
        assert outcome.calls[0].status_code == 200
        assert outcome.calls[0].provider_ttft_ms is not None
        # Immediate external cancellation can share a Windows clock tick with dispatch.
        assert outcome.stage_ms["draft"] >= 0
        assert outcome.tokens_in is None and not outcome.usage.complete
        row = ledger.get_call(outcome.calls[0].call_id)
        assert row.status == "timeout" and row.tokens_in is None


@pytest.mark.parametrize("stage", ["draft", "verification"])
async def test_deadline_cancels_actual_p3_service_and_starts_no_further_calls(
    tmp_path: Path, loopback_ports: set[int], stage: str
) -> None:
    engine = fixture_engine()  # Warm the existing offline tokenizer outside the request budget.
    p = profile(faults__timeout_prob=1 if stage == "draft" else 0)
    if stage == "verification":
        data = p.model_dump()
        for kind in data["timing"].values():
            for name, value in (("ttft_ms", 1), ("decode_tps", 100000), ("completion_tokens", 100)):
                kind[name] = {
                    "family": "constant",
                    "parameters": {
                        "value": [value],
                        "source": "chosen",
                        "note": "FAKE controlled deadline fixture",
                    },
                }
        data["timing"]["verify"]["ttft_ms"]["parameters"]["value"] = [10000]
        p = Profile.model_validate(data)
    service = create_app(profile=p)
    c = SystemClock()
    async with running(service, loopback_ports) as url:
        with Ledger(tmp_path / "wire.sqlite3", clock=c) as ledger:
            async with GroqCompatibleProvider(
                url, None, p.model, ledger=ledger, clock=c
            ) as provider:
                app = Orchestrator(
                    provider,
                    ToolExecutor(engine.search_papers, clock=c),
                    model=p.model,
                    clock=c,
                    ledger=ledger,
                )
                outcome = await app.handle("alpha", setting="A", deadline_ms=400)
            await until_idle(service)
            assert outcome.state == State.TIMED_OUT and not outcome.releasable
            assert outcome.provider_calls == (1 if stage == "draft" else 2)
            assert outcome.repair_calls == 0 and outcome.first_token_ms is None
            assert not outcome.usage.complete and outcome.tokens_in is None
            assert all(ledger.get_call(a.call_id).status != "pending" for a in outcome.calls)
