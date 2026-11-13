import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from loadgen.arrivals import ArrivalConfig, plan
from loadgen.client import LoadClient, dispatch
from loadgen.models import RunContext
from loadgen.settings import load_settings
from loadgen.sse import SSEDecoder
from relay.app.api import sse
from relay.obs.clock import SystemClock
from relay.obs.events import EventWriter
from tests.standin_server import running


def completion(rid: str) -> str:
    return sse(
        "complete",
        {
            "request_id": rid,
            "status": "ok",
            "provider_calls": 0,
            "retries": 0,
            "mode": "normal",
            "cache_hit": False,
            "coalesced": False,
            "citations": [],
            "usage": {
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_cached": 0,
                "counted_tokens": 0,
                "complete": True,
            },
        },
    )


def test_first_answer_timestamp_tracks_content_byte_not_frame_completion() -> None:
    parser = SSEDecoder(1000)
    assert not parser.feed(b'event: answer\ndata: {"request_id":"r","text":"', 10)
    assert not parser.feed(b"FAKE", 20)
    frames = parser.feed(b'"}\n\n', 30)
    assert frames[0].answer_byte_ns == 20


async def test_slow_http_server_does_not_reduce_arrivals(
    tmp_path: Path, loopback_ports: set[int]
) -> None:
    app = FastAPI()
    gate = asyncio.Event()
    received = []

    @app.post("/ask")
    async def ask(request: Request) -> StreamingResponse:
        rid = request.headers["x-request-id"]
        received.append(rid)

        async def stream() -> object:
            await gate.wait()
            yield sse("metadata", {"first_token_ms": 0})
            yield sse("answer", {"request_id": rid, "text": "FAKE safe answer"})
            yield completion(rid)

        return StreamingResponse(stream(), media_type="text/event-stream")

    c = SystemClock()
    config = load_settings()
    plans = plan(
        ArrivalConfig(kind="constant", segments=[{"duration_s": 0.08, "rate_per_minute": 3000}]),
        ["fixture"],
        run_id="FAKE-open",
        seed=7,
        setting="A",
        sampling="uniform",
        maximum=10,
        exponent=1,
    )
    async with running(app, loopback_ports) as url:
        client = LoadClient(url.removesuffix("/openai/v1"), config, c)
        task = asyncio.create_task(
            dispatch(
                client,
                plans,
                {"fixture": "FAKE question"},
                RunContext(
                    run_id="FAKE",
                    scenario="fixture-only",
                    model="FAKE",
                    deadline_ms=2000,
                    cache=False,
                    fixture_only=True,
                ),
                [EventWriter(tmp_path / "rows", clock=c)],
            )
        )

        async def all_arrived() -> None:
            while len(received) < 4:
                await asyncio.sleep(0)

        await asyncio.wait_for(all_arrived(), 1)
        assert not task.done()
        gate.set()
        rows = await task
        await client.close()
    assert len(rows) == 4 and all(r.status == "ok" for r in rows)
    assert all(r.dispatch_timestamp_source == "httpcore_headers" for r in rows)
    assert all(r.first_token_ms >= 0 for r in rows)
    assert all(r.first_answer_byte_monotonic >= r.actual_send_monotonic for r in rows)
    assert "FAKE safe answer" not in (tmp_path / "rows").read_text()


@pytest.mark.parametrize("mode", ["incomplete", "wrong_id", "invalid_usage", "rejected"])
async def test_invalid_protocol_or_rejection_is_never_success(tmp_path: Path, mode: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        rid = request.headers["x-request-id"]
        if mode == "rejected":
            return httpx.Response(429, json={"reason": "queue_full"})
        text = sse(
            "answer", {"request_id": "wrong" if mode == "wrong_id" else rid, "text": "FAKE answer"}
        )
        if mode != "incomplete":
            text += (
                completion(rid).replace('"provider_calls": 0', '"provider_calls": -1')
                if mode == "invalid_usage"
                else completion(rid)
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=text)

    c = SystemClock()
    client = LoadClient(
        "http://127.0.0.1", load_settings(), c, transport=httpx.MockTransport(handler)
    )
    items = plan(
        ArrivalConfig(kind="one_shot", count=1),
        ["fixture"],
        run_id="FAKE",
        seed=1,
        setting="A",
        sampling="uniform",
        maximum=1,
        exponent=1,
    )
    rows = await dispatch(
        client,
        items,
        {"fixture": "FAKE"},
        RunContext(
            run_id="FAKE",
            scenario="fixture-only",
            model="FAKE",
            deadline_ms=1000,
            cache=False,
            fixture_only=True,
        ),
        [EventWriter(tmp_path / "rows", clock=c)],
    )
    await client.close()
    assert rows[0].status == ("rate_limited" if mode == "rejected" else "error")
    assert rows[0].first_token_ms is None
