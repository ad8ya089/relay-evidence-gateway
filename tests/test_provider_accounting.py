"""External-origin cases use synthetic credentials and MockTransport, never Groq."""

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from relay.control.ledger import Ledger
from relay.obs.events import EventWriter
from relay.providers.base import CallMetadata, ProviderError
from relay.providers.groq import GroqCompatibleProvider, is_loopback_url
from relay.providers.guard import RealCallGuard
from relay.settings import SafetySettings
from tests.fake_groq_server import FakeGroqServer, completion
from tests.test_provider_http import request

EXTERNAL = "https://api.groq.com/openai/v1"
FAKE_KEY = "FAKE-key-only-for-in-memory-mocked-transport"


def safety(ledger: Ledger, *, cap: int = 100, request_cap: int = 2) -> RealCallGuard:
    return RealCallGuard(
        SafetySettings(allow_real="1", real_token_cap=cap, real_request_cap=request_cap),
        argv=["--real"],
        ledger=ledger,
        session_id="simulated-session",
    )


@pytest.mark.parametrize(
    "url", [EXTERNAL, "https://127.0.0.1.example/openai/v1", "http://192.0.2.1/v1"]
)
async def test_external_origins_fail_closed_before_transport(url: str) -> None:
    count = 0

    def forbidden(req: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json=completion())

    async with GroqCompatibleProvider(
        url, None, "FAKE", transport=httpx.MockTransport(forbidden)
    ) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request(token_reservation=100))
    assert raised.value.kind == "budget_exceeded" and count == 0
    assert not raised.value.info.safe_context["dispatched"]


async def test_simulated_external_attempt_reserves_records_and_reconciles(tmp_path: Path) -> None:
    count = 0

    def serve(req: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json=completion())

    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        guard = safety(ledger)
        async with GroqCompatibleProvider(
            EXTERNAL,
            FAKE_KEY,
            "FAKE-model",
            ledger=ledger,
            guard=guard,
            transport=httpx.MockTransport(serve),
        ) as provider:
            result = await provider.complete(request(token_reservation=100))
            assert result.usage.counted_tokens == guard.tokens_reserved == 90
            with pytest.raises(ProviderError, match="budget_exceeded"):
                await provider.complete(request(token_reservation=100))
        assert count == ledger.requests_used_today("FAKE-model") == 1
        assert ledger.session_usage(guard.session_id).unknown_usage_count == 0


async def test_inflight_reservation_blocks_second_adapter(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def serve(req: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, json=completion())

    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        one, two = safety(ledger), safety(ledger)
        async with (
            GroqCompatibleProvider(
                EXTERNAL,
                FAKE_KEY,
                "FAKE-model",
                ledger=ledger,
                guard=one,
                transport=httpx.MockTransport(serve),
            ) as first,
            GroqCompatibleProvider(
                EXTERNAL,
                FAKE_KEY,
                "other-model",
                ledger=ledger,
                guard=two,
                transport=httpx.MockTransport(serve),
            ) as second,
        ):
            pending = asyncio.create_task(first.complete(request(token_reservation=80)))
            await asyncio.wait_for(entered.wait(), 2)
            try:
                with pytest.raises(ProviderError, match="budget_exceeded"):
                    await second.complete(request(token_reservation=32))
                assert ledger.session_usage(one.session_id).inflight_attempts == 1
            finally:
                release.set()
                await pending
    assert count == 1


async def test_unknown_429_is_not_free_usage(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        guard = safety(ledger)
        async with GroqCompatibleProvider(
            EXTERNAL,
            FAKE_KEY,
            "FAKE-model",
            ledger=ledger,
            guard=guard,
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    429,
                    json={"error": {"message": "FAKE unrecognised"}},
                    headers={"retry-after": "2"},
                )
            ),
        ) as provider:
            req = request(token_reservation=100)
            with pytest.raises(ProviderError) as raised:
                await provider.complete(req)
        assert raised.value.kind == "rate_limit_unknown"
        row = ledger.get_call(req.metadata.call_id)
        assert row.status == "rate_limited" and row.status_code == 429 and row.tokens_in is None
        assert guard.tokens_reserved == 100 and ledger.requests_used_today("FAKE-model") == 1


async def test_explicit_retry_is_a_new_record(fake_server: FakeGroqServer, tmp_path: Path) -> None:
    fake_server.mode = "server"
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        async with GroqCompatibleProvider(
            fake_server.base_url, None, "FAKE", ledger=ledger, session_id="local"
        ) as provider:
            for attempt in (1, 2):
                with pytest.raises(ProviderError):
                    await provider.complete(
                        request(
                            metadata=CallMetadata(
                                request_id="logical",
                                call_id=f"attempt{attempt}",
                                attempt=attempt,
                            )
                        )
                    )
        assert ledger.session_usage("local", real=False).attempts == 2
        assert not ledger.get_call("attempt1").real and not ledger.get_call("attempt2").real
        assert ledger.requests_used_today("FAKE") == 0
    assert fake_server.request_count == 2


async def test_validation_failure_retains_reported_usage(tmp_path: Path) -> None:
    class DifferentSchema(BaseModel):
        required_other_field: str

    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        guard = safety(ledger)
        async with GroqCompatibleProvider(
            EXTERNAL,
            FAKE_KEY,
            "FAKE",
            ledger=ledger,
            guard=guard,
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion())),
        ) as provider:
            req = request(token_reservation=100, validation_model=DifferentSchema)
            with pytest.raises(ProviderError) as raised:
                await provider.complete(req)
        assert (
            raised.value.kind == "invalid_response" and raised.value.info.usage.counted_tokens == 90
        )
        assert ledger.get_call(req.metadata.call_id).counted_tokens == 90


async def test_cancellation_is_recorded_and_closes_http(
    fake_server: FakeGroqServer, tmp_path: Path
) -> None:
    fake_server.mode = "hang"
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        async with GroqCompatibleProvider(
            fake_server.base_url, None, "FAKE", ledger=ledger
        ) as provider:
            req = request(stream=True, token_reservation=100)
            pending = asyncio.create_task(provider.complete(req))
            await asyncio.wait_for(fake_server.entered.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await asyncio.wait_for(fake_server.disconnected.wait(), 2)
        row = ledger.get_call(req.metadata.call_id)
        assert row.error_kind == "cancelled" and row.status == "timeout" and row.tokens_in is None


async def test_broken_event_persistence_prevents_dispatch(tmp_path: Path) -> None:
    count = 0

    def forbidden(req: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json=completion())

    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        "FAKE",
        events=EventWriter(tmp_path),
        transport=httpx.MockTransport(forbidden),
    ) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request())
    assert raised.value.kind == "persistence" and count == 0


def test_credential_urls_are_rejected_and_localhost_is_exact() -> None:
    for url in ("https://user:FAKE@example.test/v1", "https://api.groq.com/v1?key=FAKE"):
        with pytest.raises(ValueError):
            is_loopback_url(url)
    assert is_loopback_url("http://localhost:8081/openai/v1")
    assert not is_loopback_url("http://localhost.example/openai/v1")


async def test_async_literal_external_ip_is_blocked_by_test_guard() -> None:
    with pytest.raises(AssertionError, match="Network is forbidden"):
        await asyncio.get_running_loop().create_connection(asyncio.Protocol, "192.0.2.1", 443)
