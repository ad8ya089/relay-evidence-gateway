import asyncio
from pathlib import Path

import httpx
import pytest

from relay.app.api import create_app, create_local_app
from relay.app.cache import cache_key
from relay.app.calls import CallRunner, Invocation, RequestCalls
from relay.app.models import Draft, Question
from relay.app.settings import pipeline_settings
from relay.control.admission import Admission, AdmissionRejected
from relay.control.breaker import CircuitBreaker
from relay.control.ledger import CallRecord, Ledger
from relay.control.limiter import Limiter, Need
from relay.control.retry import NaiveRetryPolicy, RetryBudget, RetryController
from relay.control.settings import load_policy
from relay.providers.base import (
    HeaderSnapshot,
    LLMRequest,
    LLMResult,
    Message,
    ProviderError,
    ProviderFailure,
)
from standin.profile import Profile
from standin.service import create_app as standin_app
from tests.fixtures.app_provider import ScriptedProvider
from tests.fixtures.control import controlled
from tests.standin_helpers import clock, profile
from tests.standin_server import running
from tests.test_control_limiter import MODEL, policy


async def test_locally_blocked_attempts_cannot_refill_retry_budget() -> None:
    c = clock()
    config = load_policy()
    breaker = CircuitBreaker(config.breaker, c)
    for _ in range(5):
        await breaker.record(MODEL, "r", "server")
    budget = RetryBudget(config.retry)
    runner = CallRunner(
        ScriptedProvider([]),
        MODEL,
        clock=c,
        config=pipeline_settings(),
        execution=RetryController(config.retry, c, budget, breaker=breaker),
    )
    book = RequestCalls(request_id="r", run_id="FAKE", arrival_ns=0, deadline_ns=10**12)
    await runner.invoke(book, "draft", [Message(role="user", content="FAKE")], Draft)
    assert (await budget.snapshot())["original_attempts"] == 0
    assert not book.attempts


async def test_naive_demo_retries_completed_server_failure_explicitly() -> None:
    calls = []

    async def operation(attempt: int) -> Invocation:
        calls.append(attempt)
        return Invocation(failure=ProviderFailure(kind="server")) if attempt == 1 else Invocation()

    result = await NaiveRetryPolicy(load_policy().retry).run(operation)
    assert result.failure is None and calls == [1, 2]


async def test_retry_limit_is_shared_across_draft_and_verifier() -> None:
    c = clock()
    config = load_policy().retry
    budget = RetryBudget(config)
    for _ in range(10):
        await budget.original()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    failure = ProviderError(ProviderFailure(kind="server"))
    provider = ScriptedProvider([failure, "FAKE", failure])
    runner = CallRunner(
        provider,
        MODEL,
        clock=c,
        config=pipeline_settings(),
        execution=RetryController(config, c, budget, sleep=sleep),
    )
    book = RequestCalls(request_id="r", run_id="FAKE", arrival_ns=0, deadline_ns=10**12)
    for kind in ("draft", "verify"):
        await runner.invoke(book, kind, [Message(role="user", content="FAKE")], Draft)
    assert len(provider.requests) == 3 and book.retries == 1


async def test_restart_preserves_daily_usage_and_conservatively_guards_minute_window(
    tmp_path: Path,
) -> None:
    c = clock()
    c.set_utc(c.utc_now().replace(hour=12))
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        ledger.record_call(
            CallRecord(
                id="prior",
                run_id="FAKE",
                kind="draft",
                model=MODEL,
                status="ok",
                real=False,
                session_id="FAKE",
                request_id="old",
                tokens_in=50,
                tokens_out=0,
                tokens_cached=0,
                total_tokens=50,
            )
        )
        limiter = Limiter(policy(), ledger, c)
        assert (await limiter.remaining(MODEL)).daily_used_tokens == 50
        assert await limiter.projected_wait(Need(model=MODEL, tokens=10, requests=1)) == 60000
        c.advance_ms(60000)
        assert await limiter.projected_wait(Need(model=MODEL, tokens=10, requests=1)) == 0


async def test_concurrent_header_delivery_preserves_newer_lower_observation(tmp_path: Path) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(policy(), ledger, c)
        gate = asyncio.Event()

        async def old() -> None:
            await gate.wait()
            await limiter.update_from_headers(
                MODEL,
                HeaderSnapshot(observed_ns=0, limit_tokens_tpm=1000, remaining_tokens_tpm=900),
            )

        async def new() -> None:
            c.advance_ms(1)
            await limiter.update_from_headers(
                MODEL,
                HeaderSnapshot(
                    observed_ns=c.monotonic_ns(), limit_tokens_tpm=50, remaining_tokens_tpm=10
                ),
            )
            gate.set()

        await asyncio.gather(old(), new())
        view = await limiter.remaining(MODEL)
        assert view.limits.tpm == 50 and view.tpm == 10


async def test_queued_reservation_is_reevaluated_after_header_drop(tmp_path: Path) -> None:
    c = clock()
    config = policy(max_active_requests=1, queue_capacity=1)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(config, ledger, c)
        admission = Admission(config, limiter, CircuitBreaker(config.breaker, c), c)
        await admission.admit("active", "A", MODEL, 10**12)
        queued = await admission.admit("queued", "A", MODEL, 10**12)
        task = asyncio.create_task(admission.enter(queued))
        await asyncio.sleep(0)
        await limiter.update_from_headers(
            MODEL, HeaderSnapshot(observed_ns=0, remaining_tokens_tpm=0)
        )
        await admission.wake()
        with pytest.raises(AdmissionRejected):
            await asyncio.wait_for(task, 1)
        await admission.release("active")
        await admission.release("queued")
        assert not limiter.reservations and ledger.daily_usage(MODEL, real=False).attempts == 0


@pytest.mark.parametrize("pressure", [False, True])
async def test_policy_caps_and_pressure_evidence_limit_reach_actual_provider(
    tmp_path: Path, pressure: bool
) -> None:
    async with controlled(tmp_path, reject=True) as plane:
        actual = plane.provider
        seen = []

        class Observed:
            async def complete(self, request: LLMRequest) -> LLMResult:
                seen.append(
                    (
                        request.metadata.kind,
                        request.max_completion_tokens,
                        sum(m.content.count("<<chunk:") for m in request.messages),
                    )
                )
                return await actual.complete(request)

        plane.provider = Observed()
        if pressure:
            await plane.limiter.update_from_headers(
                MODEL, HeaderSnapshot(observed_ns=0, remaining_tokens_tpm=30000)
            )
        await plane.ask(Question(text="alpha", setting="A"))
        assert seen[0][1] == (280 if pressure else 450)
        assert all(cap == 120 for kind, cap, _ in seen if kind == "verify")
        if pressure:
            assert seen[0][2] <= 3 and len(seen) == 2
        else:
            assert len(seen) == 4


async def test_api_state_is_local_and_invalid_id_does_not_dispatch(tmp_path: Path) -> None:
    async with controlled(tmp_path) as plane:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane), client=("203.0.113.1", 1234)),
            base_url="http://relay.local",
        ) as client:
            state = await client.get("/state", headers={"X-Forwarded-For": "127.0.0.1"})
            assert state.status_code == 403
            rejected = await client.post(
                "/ask",
                json={"question": "alpha", "setting": "A"},
                headers={"X-Request-Id": "bad id"},
            )
            assert rejected.status_code == 422
            assert plane.ledger.daily_usage(MODEL, real=False).attempts == 0


def test_factory_rejects_external_origin_before_creating_accounting_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_local_app(
            base_url="https://api.groq.com/openai/v1",
            ledger_path=tmp_path / "ledger",
            cache_path=tmp_path / "cache",
        )
    assert not (tmp_path / "ledger").exists() and not (tmp_path / "cache").exists()


async def test_pending_reservation_survives_utc_rollover_and_restart(tmp_path: Path) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(policy(), ledger, c)
        need = Need(model=MODEL, tokens=100, requests=1)
        assert await limiter.reserve("r", need)
        await limiter.start_call("r", "pending", need)
        ledger.record_call(
            CallRecord(
                id="pending",
                run_id="FAKE",
                kind="draft",
                model=MODEL,
                status="pending",
                real=False,
                session_id="FAKE",
                request_id="r",
                reserved_tokens=100,
            )
        )
        c.advance_ms(60000)
        await limiter.release("r")
        assert (await limiter.remaining(MODEL)).daily_used_tokens == 100
        restarted = Limiter(policy(), ledger, c)
        c.advance_ms(60000)
        view = await restarted.remaining(MODEL)
        assert view.daily_used_tokens == 100 and view.outstanding_tokens == 100 and view.uncertain


async def test_restart_at_utc_boundary_cannot_erase_previous_minute_activity(
    tmp_path: Path,
) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        ledger.record_call(
            CallRecord(
                id="old",
                run_id="FAKE",
                kind="draft",
                model=MODEL,
                status="ok",
                real=False,
                session_id="FAKE",
                request_id="r",
                tokens_in=1,
                tokens_out=1,
                tokens_cached=0,
                total_tokens=2,
            )
        )
        c.advance_ms(30000)
        limiter = Limiter(policy(), ledger, c)
        assert (await limiter.remaining(MODEL)).daily_used_tokens == 0
        assert await limiter.projected_wait(Need(model=MODEL, tokens=1, requests=1)) == 60000


async def test_expired_queue_service_estimate_is_rejected_before_execution(tmp_path: Path) -> None:
    c = clock()
    config = policy(max_active_requests=1, queue_capacity=1, max_queue_wait_ms=4000)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        admission = Admission(
            config, Limiter(config, ledger, c), CircuitBreaker(config.breaker, c), c
        )
        await admission.admit("active", "A", MODEL, 10**12)
        queued = await admission.admit("queued", "A", MODEL, 6_000_000_000)
        c.advance_ms(3500)
        await admission.release("active")
        with pytest.raises(AdmissionRejected) as rejected:
            await admission.enter(queued)
        assert rejected.value.rejection.reason == "deadline_infeasible"
        await admission.release("queued")


async def test_assumed_slot_completion_does_not_create_retry_after_hint(tmp_path: Path) -> None:
    c = clock()
    config = policy(max_active_requests=1, queue_capacity=1, max_queue_wait_ms=1000)
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        admission = Admission(
            config, Limiter(config, ledger, c), CircuitBreaker(config.breaker, c), c
        )
        await admission.admit("active", "A", MODEL, 10**12)
        rejected = await admission.admit("queued", "A", MODEL, 10**12)
        assert rejected.reason == "projected_wait" and rejected.retry_after_s is None
        await admission.release("active")


async def test_unknown_quota_cannot_borrow_a_daily_request_reset_hint(tmp_path: Path) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(policy(), ledger, c)
        await limiter.update_from_headers(
            MODEL,
            HeaderSnapshot(
                observed_ns=0,
                remaining_requests_rpd=10,
                headers={"x-ratelimit-reset-requests": "1h"},
            ),
        )
        await limiter.on_quota_error(MODEL, "rate_limit_unknown", 1000)
        assert (await limiter.remaining(MODEL)).retry_after_s is None


async def test_backend_exception_is_a_safe_503_without_internal_text(tmp_path: Path) -> None:
    async with controlled(tmp_path) as plane:

        async def broken(question: Question, **kwargs: object) -> object:
            raise RuntimeError("FAKE internal stack detail")

        plane.ask = broken
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post("/ask", json={"question": "alpha", "setting": "A"})
            assert response.status_code == 503 and "internal stack" not in response.text


async def test_zero_observed_limit_disables_capacity_without_division_by_zero(
    tmp_path: Path,
) -> None:
    c = clock()
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(policy(), ledger, c)
        await limiter.update_from_headers(MODEL, HeaderSnapshot(observed_ns=0, limit_tokens_tpm=0))
        assert (await limiter.remaining(MODEL)).limits.tpm == 0
        assert await limiter.projected_wait(Need(model=MODEL, tokens=1, requests=1)) is None
        assert (await limiter.utilisation(MODEL))["minute"] == 1


async def test_naive_policy_requires_opt_in_and_limits_retries_across_calls() -> None:
    c = clock()
    config = load_policy().retry.model_copy(update={"policy": "naive_s7a"})
    budget = RetryBudget(config)
    with pytest.raises(ValueError):
        RetryController(config, c, budget)
    failure = ProviderError(ProviderFailure(kind="server"))
    provider = ScriptedProvider([failure] * 5)
    runner = CallRunner(
        provider,
        MODEL,
        clock=c,
        config=pipeline_settings(),
        execution=RetryController(config, c, budget, allow_unsafe_naive=True),
    )
    book = RequestCalls(request_id="r", run_id="FAKE", arrival_ns=0, deadline_ns=10**12)
    for kind in ("draft", "verify"):
        await runner.invoke(book, kind, [Message(role="user", content="FAKE")], Draft)
    assert book.retries == 3 and len(provider.requests) == 5


async def test_rpm_remains_local_and_rolling_independently_of_rpd_headers(tmp_path: Path) -> None:
    c = clock()
    config = policy(limits={MODEL: {"rpm": 1, "tpm": 100, "rpd": 10, "tpd": 1000}})
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(config, ledger, c)
        need = Need(model=MODEL, tokens=10, requests=1)
        assert await limiter.reserve("r", need)
        await limiter.start_call("r", "call", need)
        ledger.record_call(
            CallRecord(
                id="call",
                run_id="FAKE",
                kind="draft",
                model=MODEL,
                status="ok",
                real=False,
                session_id="FAKE",
                request_id="r",
                tokens_in=10,
                tokens_out=0,
                tokens_cached=0,
                total_tokens=10,
            )
        )
        from relay.providers.base import Usage

        await limiter.reconcile(
            "call", Usage(tokens_in=10, tokens_out=0, tokens_cached=0, total_tokens=10)
        )
        await limiter.release("r")
        await limiter.update_from_headers(
            MODEL,
            HeaderSnapshot(observed_ns=0, limit_requests_rpd=1000, remaining_requests_rpd=999),
        )
        assert (await limiter.remaining(MODEL)).rpm == 0
        assert (await limiter.remaining(MODEL)).tpm == 90
        assert await limiter.projected_wait(need) == 60000
        c.advance_ms(60000)
        assert (await limiter.remaining(MODEL)).rpm == 1


async def test_validation_abstention_never_populates_enabled_cache(tmp_path: Path) -> None:
    cache = policy().cache.model_dump() | {"enabled": True}
    async with controlled(tmp_path, changes={"cache": cache}, reject=True) as plane:
        outcome = await plane.ask(Question(text="alpha", setting="A"))
        assert outcome.status == "error" and outcome.releasable
        assert (await plane.cache.snapshot())["writes"] == 0


async def test_corrupted_persistent_cache_entry_cannot_restore_approval(tmp_path: Path) -> None:
    cache = policy().cache.model_dump() | {"enabled": True}
    async with controlled(tmp_path, changes={"cache": cache}) as plane:
        await plane.ask(Question(text="alpha", setting="A"))
        key = cache_key("alpha", "A", MODEL, plane.policy.cache.prompt_version, plane.corpus_hash)
        plane.cache.entries.clear()
        plane.cache.db.execute("UPDATE answers SET payload=payload || ' '")
        assert await plane.cache.get(key) is None


async def test_durable_excess_is_respected_before_async_reconciliation_resumes(
    tmp_path: Path,
) -> None:
    c = clock()
    config = policy(limits={MODEL: {"rpm": 10, "tpm": 100, "rpd": 100, "tpd": 1000}})
    with Ledger(tmp_path / "ledger", clock=c) as ledger:
        limiter = Limiter(config, ledger, c)
        need = Need(model=MODEL, tokens=40, requests=1)
        assert await limiter.reserve("r", need)
        await limiter.start_call("r", "call", need)
        ledger.record_call(
            CallRecord(
                id="call",
                run_id="FAKE",
                kind="draft",
                model=MODEL,
                status="ok",
                real=False,
                session_id="FAKE",
                request_id="r",
                tokens_in=80,
                tokens_out=10,
                tokens_cached=0,
                total_tokens=90,
            )
        )
        assert (await limiter.remaining(MODEL)).tpm == 10
        assert not await limiter.reserve("other", Need(model=MODEL, tokens=50, requests=1))


@pytest.mark.parametrize(
    "scope,reason,calls", [("rpm", "provider_429_minute", 2), ("tpd", "provider_429_day", 1)]
)
async def test_actual_p3_quota_codes_have_explicit_scope_and_known_retry_hint(
    tmp_path: Path, loopback_ports: set[int], scope: str, reason: str, calls: int
) -> None:
    if not Path("data/corpus_manifest.json").exists():
        pytest.skip("Pinned corpus requires P1 ingestion")
    data = profile().model_dump()
    data["models"][MODEL][scope]["value"] = 1
    async with running(
        standin_app(profile=Profile.model_validate(data)), loopback_ports
    ) as model_url:
        app = create_local_app(
            base_url=model_url,
            no_cache=True,
            ledger_path=tmp_path / "ledger",
            cache_path=tmp_path / "cache",
            events_path=tmp_path / "events",
        )
        async with app.router.lifespan_context(app):
            async with running(app, loopback_ports) as url:
                async with httpx.AsyncClient(
                    base_url=url.removesuffix("/openai/v1"), trust_env=False, timeout=15
                ) as client:
                    payload = {"question": "Explain encoder-decoder attention", "setting": "A"}
                    response = await client.post("/ask", json=payload)
                    assert response.status_code == 429 and response.json()["reason"] == reason, (
                        response.text
                    )
                    assert "retry-after" in response.headers and "retry_after_s" in response.json()
                    assert app.state.control.ledger.daily_usage(MODEL, real=False).attempts == calls
                    await client.post("/ask", json=payload)
                    assert app.state.control.ledger.daily_usage(MODEL, real=False).attempts == calls
