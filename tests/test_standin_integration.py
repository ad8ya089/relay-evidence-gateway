import asyncio
import json
from pathlib import Path

import httpx
import pytest

from relay.obs.clock import FakeClock
from relay.obs.events import EventWriter
from relay.providers.base import LLMRequest, Message, ProviderError, ResponseFormat
from relay.providers.groq import GroqCompatibleProvider
from standin.generation import Draft, Verification
from standin.profile import Profile
from standin.service import create_app
from standin.simulation import Simulation
from tests.standin_helpers import clock, profile
from tests.test_standin_generation import messages
from tests.test_standin_service import request


def constant_profile(**overrides: object) -> Profile:
    data = profile(**overrides).model_dump()
    for kind in data["timing"].values():
        for name, value in (("ttft_ms", 100), ("decode_tps", 100), ("completion_tokens", 100)):
            kind[name] = {
                "family": "constant",
                "parameters": {
                    "value": [value],
                    "source": "chosen",
                    "note": "Controlled test fixture",
                },
            }
    return Profile.model_validate(data)


async def test_draft_verify_cache_and_safe_provider_events(tmp_path: Path) -> None:
    c, p = clock(), profile(faults__verify_fail_prob=1)

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=p, clock=c, sleep=sleep)
    path = tmp_path / "events.jsonl"
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        p.model,
        clock=c,
        events=EventWriter(path, clock=c),
        transport=httpx.ASGITransport(app=app),
    ) as provider:
        draft_req = LLMRequest(
            messages=messages(
                "draft",
                "Question\n<<chunk:real-supplied-id>>Evidence sentence with 42 units.<<end>>",
            ),
            max_completion_tokens=450,
            validation_model=Draft,
            response_format=ResponseFormat(type="json_object"),
        )
        first = await provider.complete(draft_req)
        second = await provider.complete(draft_req.model_copy(update={"stream": True}))
        assert first.text == second.text
        assert first.usage.tokens_cached == 0 and second.usage.tokens_cached > 0
        assert (
            second.structured_output["claims"][0]["evidence"][0]["chunk_id"] == "real-supplied-id"
        )
        verify_req = LLMRequest(
            messages=messages(
                "verify",
                json.dumps(
                    [
                        {"i": 0, "claim": "Evidence", "quotes": ["Evidence"]},
                        {"i": 1, "claim": "Other", "quotes": ["Other"]},
                    ]
                ),
            ),
            max_completion_tokens=250,
            stream=True,
            validation_model=Verification,
        )
        verified = await provider.complete(verify_req)
        assert [v["i"] for v in verified.structured_output["verdicts"]] == [0, 1]
        assert [v["v"] for v in verified.structured_output["verdicts"]] == [
            "unsupported",
            "supported",
        ]
    state = app.state.simulation.snapshot()
    assert state["models"][p.model]["usage"]["tpm"] == sum(
        r.usage.counted_tokens for r in (first, second, verified)
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row.get("source", "stand-in") == "stand-in" for row in rows)
    assert "authorization" not in path.read_text().casefold()
    assert "Evidence sentence" not in path.read_text()


async def test_slow_success_is_success_with_multiplied_measured_duration() -> None:
    durations = []
    for slow in (0, 1):
        c = clock()
        p = constant_profile(faults__slow_success_prob=slow, faults__slow_factor=8)

        async def sleep(seconds: float, test_clock: FakeClock = c) -> None:
            test_clock.advance_ms(seconds * 1000)

        app = create_app(profile=p, clock=c, sleep=sleep)
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            p.model,
            clock=c,
            transport=httpx.ASGITransport(app=app),
        ) as provider:
            result = await provider.complete(request())
        assert result.status_code == 200
        durations.append(result.total_ms)
    assert durations[1] == pytest.approx(durations[0] * 8, abs=1e-4)


async def test_six_overlapping_lifetimes_follow_exact_load_formula() -> None:
    p, c = constant_profile(capacity=2, alpha=0.5), clock()
    baseline = Simulation(p, c)
    one = baseline.begin(request())
    single = one.decision.ttft_ms + one.decision.decode_ms(one.tokens_out)
    baseline.finish(one, success=True)
    burst = Simulation(p, c)
    prepared = [burst.begin(request()) for _ in range(6)]
    assert burst.inflight == 6
    totals = []
    for i, call in enumerate(prepared, 1):
        multiplier = 1 + 0.5 * max(0, (i - 2) / 2)
        total = call.decision.ttft_ms + call.decision.decode_ms(call.tokens_out)
        assert total == pytest.approx(single * multiplier)
        totals.append(total)
    assert sum(totals) / len(totals) > single
    for call in prepared:
        burst.finish(call, success=True)
    assert burst.inflight == 0 and burst.quotas.view(p.model, p.models[p.model]).pending_tokens == 0
    entered, release = asyncio.Event(), asyncio.Event()
    virtual = clock()

    async def sleep(seconds: float) -> None:
        if app.state.simulation.inflight == 6:
            entered.set()
        await release.wait()
        virtual.advance_ms(seconds * 1000)
        await asyncio.sleep(0)

    app = create_app(profile=p, clock=virtual, sleep=sleep)
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        p.model,
        clock=virtual,
        transport=httpx.ASGITransport(app=app),
    ) as provider:
        calls = [asyncio.create_task(provider.complete(request())) for _ in range(6)]
        await asyncio.wait_for(entered.wait(), 2)
        assert app.state.simulation.inflight == 6
        release.set()
        results = await asyncio.gather(*calls)
    assert all(result.status_code == 200 for result in results)
    assert sum(result.total_ms for result in results) / 6 > single
    assert [d.load_multiplier for d in app.state.simulation.decisions] == [1, 1, 1.25, 1.5, 1.75, 2]
    assert app.state.simulation.inflight == 0


async def test_active_call_blocks_second_reservation_and_unsafe_reset() -> None:
    c, p = clock(), profile()
    data = p.model_dump()
    data["models"][p.model]["tpm"]["value"] = 200
    p = Profile.model_validate(data)
    entered, release = asyncio.Event(), asyncio.Event()

    async def sleep(seconds: float) -> None:
        entered.set()
        await release.wait()
        c.advance_ms(seconds * 1000)

    app = create_app(profile=p, clock=c, sleep=sleep)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        pending = asyncio.create_task(
            client.post("/openai/v1/chat/completions", json=request().to_payload(p.model))
        )
        await asyncio.wait_for(entered.wait(), 2)
        blocked = await client.post(
            "/openai/v1/chat/completions", json=request().to_payload(p.model)
        )
        assert blocked.status_code == 429 and "retry-after" not in blocked.headers
        assert (await client.post("/admin/reset")).status_code == 409
        assert app.state.simulation.inflight == 1
        assert (await client.post("/admin/limits", json={"tpm": 100})).status_code == 200
        release.set()
        completed = await pending
        assert completed.status_code == 200
        assert completed.headers["x-ratelimit-limit-tokens"] == "100"
    assert app.state.simulation.inflight == 0


@pytest.mark.parametrize(
    "kind,text", [("verify", "malformed"), ("draft", "<<chunk:x>>unterminated")]
)
async def test_malformed_protocol_inputs_fail_without_quota_leaks(kind: str, text: str) -> None:
    app = create_app(profile=profile())
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1", None, profile().model, transport=httpx.ASGITransport(app=app)
    ) as provider:
        with pytest.raises(ProviderError) as caught:
            await provider.complete(request().model_copy(update={"messages": messages(kind, text)}))
    assert caught.value.status == 400 and app.state.simulation.inflight == 0


async def test_cache_is_model_specific_through_provider() -> None:
    c, p = clock(), profile()
    data = p.model_dump()
    data["models"]["synthetic-second-model"] = data["models"][p.model]
    p = Profile.model_validate(data)

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=p, clock=c, sleep=sleep)
    call = request().model_copy(
        update={
            "messages": [
                Message(role="system", content="Exact static prefix"),
                Message(role="user", content="Q"),
            ]
        }
    )
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1", None, p.model, clock=c, transport=httpx.ASGITransport(app=app)
    ) as provider:
        first = await provider.complete(call)
        hit = await provider.complete(call)
        other = await provider.complete(call.model_copy(update={"model": "synthetic-second-model"}))
    assert first.usage.tokens_cached == other.usage.tokens_cached == 0
    assert hit.usage.tokens_cached > 0
