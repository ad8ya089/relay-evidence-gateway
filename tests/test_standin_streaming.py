import json

import httpx
import pytest

from relay.providers.base import ProviderError
from relay.providers.groq import GroqCompatibleProvider
from standin.service import create_app
from standin.simulation import Simulation
from standin.streaming import content_stream
from tests.standin_helpers import clock, profile
from tests.test_standin_service import request


async def test_stream_wire_chunks_delay_and_total_match() -> None:
    c, p = clock(), profile(chunks=7)
    sim = Simulation(p, c)
    prepared = sim.begin(request(stream=True))
    began = c.monotonic_ns()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    rows, times = [], []
    async for raw in content_stream(prepared, sim, sleep):
        times.append((c.monotonic_ns() - began) / 1e6)
        data = raw.decode().removeprefix("data: ").strip()
        rows.append(data if data == "[DONE]" else json.loads(data))
    content = [
        r["choices"][0]["delta"]["content"]
        for r in rows[:-1]
        if r["choices"] and "content" in r["choices"][0]["delta"]
    ]
    assert len(content) == 7 and "".join(content) == prepared.text
    assert times[0] == pytest.approx(prepared.decision.ttft_ms, abs=1e-5)
    assert times[-1] == pytest.approx(
        prepared.decision.ttft_ms + prepared.decision.decode_ms(prepared.tokens_out), abs=1e-5
    )
    assert rows[-2]["choices"] == [] and rows[-2]["usage"] == prepared.usage
    assert rows[-1] == "[DONE]" and sim.inflight == 0
    other = Simulation(p, clock()).begin(request())
    assert prepared.text == other.text and prepared.tokens_out == other.tokens_out


async def test_same_p2_provider_consumes_sse_without_changes() -> None:
    c = clock()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=profile(), clock=c, sleep=sleep)
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        profile().model,
        transport=httpx.ASGITransport(app=app),
        clock=c,
    ) as provider:
        result = await provider.complete(request(stream=True))
    assert result.structured_output == {"answer": "ok"}
    assert result.usage_complete and result.provider_ttft_ms is not None
    assert app.state.simulation.inflight == 0


async def test_midstream_eof_is_typed_disconnect_and_releases_state() -> None:
    c = clock()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=profile(faults__disconnect_prob=1), clock=c, sleep=sleep)
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        profile().model,
        transport=httpx.ASGITransport(app=app),
        clock=c,
    ) as provider:
        with pytest.raises(ProviderError) as caught:
            await provider.complete(request(stream=True))
    assert caught.value.kind == "disconnect"
    assert app.state.simulation.inflight == 0
