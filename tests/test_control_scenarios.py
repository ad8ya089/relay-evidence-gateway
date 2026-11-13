import asyncio
import json
from pathlib import Path

import httpx
import pytest

from relay.app.api import create_app, create_local_app
from relay.app.models import Outcome, Question, State
from relay.control.admission import Reject
from relay.providers.base import HeaderSnapshot, LLMRequest, LLMResult
from relay.providers.groq import GroqCompatibleProvider
from standin.service import create_app as standin_app
from tests.fixtures.control import controlled
from tests.standin_helpers import profile
from tests.standin_server import running, until_idle
from tests.test_app_deadlines import HangingStream
from tests.test_control_api import frames
from tests.test_control_limiter import MODEL, policy


async def until(predicate: object) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


async def test_small_burst_sheds_early_and_never_exceeds_queue_capacity(tmp_path: Path) -> None:
    async with controlled(
        tmp_path,
        changes={
            "max_active_requests": 1,
            "queue_capacity": 2,
            "max_queue_wait_ms": 6000,
            "deadline_ms": 20000,
        },
    ) as plane:
        gate, started = asyncio.Event(), asyncio.Event()
        actual = plane.provider

        class PausedProvider:
            async def complete(self, request: LLMRequest) -> LLMResult:
                started.set()
                await gate.wait()
                return await actual.complete(request)

        plane.provider = PausedProvider()
        first = asyncio.create_task(
            plane.ask(Question(text="alpha", setting="A", request_id="first"))
        )
        await started.wait()
        burst = [
            asyncio.create_task(plane.ask(Question(text="alpha", setting="A", request_id=str(i))))
            for i in range(20)
        ]
        await until(lambda: sum(t.done() for t in burst) >= 18)
        assert (await plane.admission.snapshot())["peak_queue_length"] == 2
        assert plane.ledger.daily_usage(plane.model, real=False).attempts == 0
        gate.set()
        outcomes = await asyncio.gather(first, *burst)
        assert sum(isinstance(o, Reject) for o in outcomes) == 18
        assert sum(isinstance(o, Outcome) and o.status == "ok" for o in outcomes) == 3
        assert plane.ledger.daily_usage(plane.model, real=False).attempts == 6
        assert not plane.limiter.reservations


async def test_conditional_repair_cannot_spend_initial_only_daily_budget(tmp_path: Path) -> None:
    limits = policy().limits[MODEL].model_dump() | {"tpd": 5000}
    async with controlled(tmp_path, changes={"limits": {MODEL: limits}}, reject=True) as plane:
        outcome = await plane.ask(Question(text="alpha", setting="A"))
        assert (
            outcome.state == State.ABSTAINED and outcome.failure_reason == "repair_budget_exhausted"
        )
        assert outcome.provider_calls == 2 and outcome.repair_calls == 0


async def test_no_cache_bypasses_both_cache_and_singleflight(tmp_path: Path) -> None:
    cache = policy().cache.model_dump() | {"enabled": True}
    async with controlled(tmp_path, changes={"cache": cache, "deadline_ms": 20000}) as plane:
        outcomes = await asyncio.gather(
            *(
                plane.ask(
                    Question(text="alpha", setting="A", request_id=str(i)), cache_enabled=False
                )
                for i in range(2)
            )
        )
        assert all(o.status == "ok" for o in outcomes)
        assert plane.ledger.daily_usage(MODEL, real=False).attempts == 4
        assert (await plane.cache.snapshot())["writes"] == 0
        assert (await plane.singleflight.snapshot())["joins"] == 0


async def test_http_owner_cancellation_preserves_surviving_singleflight_waiter(
    tmp_path: Path,
) -> None:
    cache = policy().cache.model_dump() | {"enabled": True}
    async with controlled(tmp_path, changes={"cache": cache}) as plane:
        gate, started = asyncio.Event(), asyncio.Event()
        actual = plane.provider

        class PausedProvider:
            async def complete(self, request: LLMRequest) -> LLMResult:
                started.set()
                await gate.wait()
                return await actual.complete(request)

        plane.provider = PausedProvider()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            owner = asyncio.create_task(
                client.post(
                    "/ask",
                    json={"question": "alpha", "setting": "A"},
                    headers={"X-Request-Id": "owner"},
                )
            )
            await started.wait()
            waiter = asyncio.create_task(
                client.post(
                    "/ask",
                    json={"question": "alpha", "setting": "A"},
                    headers={"X-Request-Id": "waiter"},
                )
            )
            await until(lambda: any(f.waiters == 2 for f in plane.singleflight.flights.values()))
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            gate.set()
            response = await waiter
            complete = frames(response.text)[-1][1]
            assert response.status_code == 200 and complete["coalesced"]
            assert complete["request_id"] == "waiter" and complete["provider_calls"] == 0
        assert plane.ledger.daily_usage(MODEL, real=False).attempts == 2
        assert (await plane.cache.snapshot())["writes"] == 1


async def test_cancelled_partial_stream_retains_charge_and_cleans_all_inflight_state(
    tmp_path: Path,
) -> None:
    async with controlled(tmp_path) as plane:
        stream = HangingStream()

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "x-ratelimit-limit-tokens": "1000",
                    "x-ratelimit-remaining-tokens": "10",
                },
                stream=stream,
            )

        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            MODEL,
            ledger=plane.ledger,
            clock=plane.clock,
            events=plane.events,
            transport=httpx.MockTransport(handler),
        ) as provider:
            plane.provider = provider
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
            ) as client:
                task = asyncio.create_task(
                    client.post("/ask", json={"question": "alpha", "setting": "A"})
                )
                await stream.entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert stream.closed.is_set()
            assert not plane.limiter.reservations
            view = await plane.limiter.remaining(MODEL)
            assert view.uncertain and view.daily_used_tokens > 0
            assert view.limits.tpm == 1000 and view.tpm <= 10
            assert plane.ledger.daily_usage(MODEL, real=False).inflight_attempts == 0
            assert (await plane.scheduler.snapshot()).inflight == 0
            assert (await plane.admission.snapshot())["active_logical_requests"] == 0


async def test_one_http_500_retry_keeps_every_p2_ledger_attempt(tmp_path: Path) -> None:
    async with controlled(tmp_path) as plane:
        delegate = plane.provider._client._transport
        calls = 0

        class FirstError(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return httpx.Response(
                        500,
                        json={
                            "error": {"message": "FAKE server"},
                            "usage": {
                                "prompt_tokens": 23,
                                "completion_tokens": 7,
                                "total_tokens": 30,
                                "prompt_tokens_details": {"cached_tokens": 3},
                            },
                        },
                    )
                return await delegate.handle_async_request(request)

        for _ in range(10):
            await plane.retry_budget.original()
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            MODEL,
            ledger=plane.ledger,
            clock=plane.clock,
            events=plane.events,
            transport=FirstError(),
        ) as provider:
            plane.provider = provider
            outcome = await plane.ask(Question(text="alpha", setting="A"))
            assert outcome.state == State.RELEASED and outcome.retries == 1 and calls == 3
            rows = [plane.ledger.get_call(a.call_id) for a in outcome.calls]
            assert rows[0].tokens_out == 7 and rows[0].status_code == 500
            assert sum(r.counted_tokens for r in rows) == outcome.counted_tokens
            assert plane.ledger.daily_usage(MODEL, real=False).attempts == 3


async def test_native_http_disconnect_cancels_p3_and_settles_reservations(
    tmp_path: Path, loopback_ports: set[int]
) -> None:
    if not Path("data/corpus_manifest.json").exists():
        pytest.skip("Pinned corpus requires P1 ingestion")
    model = standin_app(profile=profile(faults__timeout_prob=1))
    async with running(model, loopback_ports) as url:
        app = create_local_app(
            base_url=url,
            no_cache=True,
            ledger_path=tmp_path / "ledger",
            cache_path=tmp_path / "cache",
            events_path=tmp_path / "events",
        )
        async with app.router.lifespan_context(app):
            plane = app.state.control
            async with running(app, loopback_ports) as relay_url:
                async with httpx.AsyncClient(
                    base_url=relay_url.removesuffix("/openai/v1"), trust_env=False, timeout=15
                ) as client:
                    task = asyncio.create_task(
                        client.post(
                            "/ask",
                            json={"question": "Explain encoder-decoder attention", "setting": "A"},
                        )
                    )
                    await until(lambda: model.state.simulation.inflight == 1)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    await until(lambda: not plane.admission.active)
                    await until_idle(model)
                    assert not plane.limiter.reservations
                    assert plane.ledger.daily_usage(MODEL, real=False).inflight_attempts == 0
                    assert (await plane.scheduler.snapshot()).inflight == 0
    events = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
    assert not any(e["ev"] == "first_user_token" for e in events)


async def test_native_http_cache_exhaustion_breaker_pressure_and_burst(
    tmp_path: Path, loopback_ports: set[int]
) -> None:
    if not Path("data/corpus_manifest.json").exists():
        pytest.skip("Pinned corpus requires P1 ingestion")
    import yaml

    config = policy(
        max_active_requests=1,
        queue_capacity=0,
        cache=policy().cache.model_dump() | {"enabled": True},
    )
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
    async with running(standin_app(profile=profile()), loopback_ports) as model_url:
        app = create_local_app(
            policy_path=path,
            base_url=model_url,
            ledger_path=tmp_path / "ledger",
            cache_path=tmp_path / "cache",
            events_path=tmp_path / "events",
        )
        async with app.router.lifespan_context(app):
            plane = app.state.control
            async with running(app, loopback_ports) as relay_url:
                async with httpx.AsyncClient(
                    base_url=relay_url.removesuffix("/openai/v1"), trust_env=False, timeout=15
                ) as client:
                    payload = {"question": "Explain encoder-decoder attention", "setting": "A"}
                    initial = await client.post("/ask", json=payload | {"cache": False})
                    assert frames(initial.text)[-1][1]["provider_calls"] == 2
                    warm = await client.post("/ask", json=payload)
                    assert frames(warm.text)[-1][1]["status"] == "ok"
                    calls = plane.ledger.daily_usage(MODEL, real=False).attempts
                    assert calls == 4
                    original = plane.limiter._limits[MODEL]
                    plane.limiter._limits[MODEL] = original.model_copy(update={"tpd": 1})
                    refused = await client.post("/ask", json=payload | {"cache": False})
                    cached = await client.post("/ask", json=payload)
                    assert (
                        refused.status_code == 429
                        and refused.json()["reason"] == "daily_budget_exhausted"
                    )
                    assert frames(cached.text)[-1][1]["cache_hit"]
                    assert plane.ledger.daily_usage(MODEL, real=False).attempts == calls
                    plane.limiter._limits[MODEL] = original
                    for _ in range(5):
                        await plane.breaker.record(MODEL, "FAKE-outage", "server")
                    rejected = await client.post("/ask", json=payload | {"cache": False})
                    assert (
                        rejected.status_code == 503 and rejected.json()["reason"] == "breaker_open"
                    )
                    assert "retry-after" in rejected.headers
                    plane.breaker.models[MODEL].until_ns = plane.clock.monotonic_ns()
                    assert await plane.breaker.allow(MODEL, "FAKE-probe")
                    await plane.breaker.record(MODEL, "FAKE-probe", None)
                    await plane.limiter.update_from_headers(
                        MODEL,
                        HeaderSnapshot(
                            observed_ns=plane.clock.monotonic_ns(),
                            limit_tokens_tpm=100000,
                            remaining_tokens_tpm=30000,
                        ),
                    )
                    pressured = await client.post("/ask", json=payload | {"cache": False})
                    assert frames(pressured.text)[-1][1]["mode"] == "pressure"
                    before = plane.ledger.daily_usage(MODEL, real=False).attempts
                    burst = await asyncio.gather(
                        *(client.post("/ask", json=payload | {"cache": False}) for _ in range(8))
                    )
                    assert sum(r.status_code == 200 for r in burst) == 1
                    assert sum(r.status_code == 429 for r in burst) == 7
                    assert plane.ledger.daily_usage(MODEL, real=False).attempts == before + 2
                    snapshot = (await client.get("/state")).json()
                    assert snapshot["queue_length"] == snapshot["provider"]["inflight"] == 0
                    print(
                        json.dumps(
                            {
                                "source": "stand-in",
                                "native_http": True,
                                "initial_cache_disabled_calls": 2,
                                "cache_hit_after_exhaustion": True,
                                "daily_rejection": 429,
                                "breaker_rejection": 503,
                                "pressure_verified": True,
                                "burst_accepted": 1,
                                "burst_rejected": 7,
                                "final_provider_calls": before + 2,
                            }
                        )
                    )
