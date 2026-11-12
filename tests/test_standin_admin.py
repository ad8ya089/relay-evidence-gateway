import httpx

from standin.service import create_app
from tests.standin_helpers import clock, profile
from tests.test_standin_service import request


async def test_hot_halving_limits_and_consistent_safe_state() -> None:
    c, p = clock(), profile()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=p, clock=c, sleep=sleep)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        old = p.models[p.model].numbers()
        result = await client.post(
            "/admin/limits", json={name: old[name] // 2 for name in ("rpm", "tpm")}
        )
        assert result.status_code == 200
        response = await client.post(
            "/openai/v1/chat/completions", json=request().to_payload(p.model)
        )
        assert response.headers["x-ratelimit-limit-tokens"] == str(old["tpm"] // 2)
        assert response.headers["x-ratelimit-limit-requests"] == str(old["rpd"])
        state = (await client.get("/admin/state")).json()
        assert state["models"][p.model]["limits"]["rpm"] == old["rpm"] // 2
        assert state["models"][p.model]["usage"]["rpm"] == 1
        assert state["inflight"] == 0 and state["last_admin_update"]["action"] == "limits"
        assert "Smoke test" not in str(state) and "authorization" not in str(state).lower()
        assert (await client.post("/admin/limits", json={"rpm": 0})).status_code == 422


async def test_reset_reproduces_initial_sequence_but_rejects_active_requests() -> None:
    app = create_app(profile=profile(), clock=clock())
    sim = app.state.simulation
    first = sim.begin(request())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.post("/admin/reset")).status_code == 409
        assert (
            await client.post("/admin/profile", json={"name": "development"})
        ).status_code == 409
        assert sim.inflight == 1
        sim.finish(first, success=True)
        assert (await client.post("/admin/reset")).status_code == 200
        again = sim.begin(request())
        assert again.decision == first.decision and again.id != first.id
        assert not again.cache.hit
        sim.finish(first, success=False)  # A delayed old stream finalizer must be harmless.
        assert sim.inflight == 1
        sim.finish(again, success=True)


async def test_profile_replacement_is_atomic_and_validated() -> None:
    app = create_app(profile=profile(), clock=clock())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.post("/admin/profile", json={"name": "free_exact"})).status_code == 400
        assert app.state.simulation.profile.name == "development"
        invalid = profile().model_dump()
        invalid["faults"]["timeout_prob"]["value"] = 2
        assert (await client.post("/admin/profile", json={"profile": invalid})).status_code == 422
        p = profile(capacity=2, seed=99)
        assert (
            await client.post("/admin/profile", json={"profile": p.model_dump()})
        ).status_code == 200
        assert app.state.simulation.profile.capacity.value == 2


async def test_admin_rejects_nonlocal_clients_and_ignores_forwarded_headers() -> None:
    app = create_app(profile=profile())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("203.0.113.1", 42)),
        base_url="http://127.0.0.1",
    ) as client:
        assert (
            await client.get("/admin/state", headers={"x-forwarded-for": "127.0.0.1"})
        ).status_code == 403
        assert (await client.post("/admin/reset")).status_code == 403
