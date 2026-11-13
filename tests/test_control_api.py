import json
from pathlib import Path

import httpx
import pytest

from relay.app.api import create_app, create_local_app
from relay.control.ledger import Ledger
from relay.providers.base import HeaderSnapshot
from standin.service import create_app as standin_app
from tests.fixtures.control import controlled
from tests.standin_helpers import profile
from tests.standin_server import running
from tests.test_control_limiter import policy


def frames(text: str) -> list[tuple[str, dict[str, object]]]:
    result = []
    for frame in text.strip().split("\n\n"):
        lines = frame.splitlines()
        result.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return result


@pytest.mark.parametrize("request_id", [None, "client-123"])
async def test_ask_sse_ids_health_and_state(tmp_path: Path, request_id: str | None) -> None:
    async with controlled(tmp_path) as plane:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post(
                "/ask",
                json={"question": "alpha", "setting": "A"},
                headers={"X-Request-Id": request_id} if request_id else {},
            )
            assert response.status_code == 200
            rid = response.headers["x-request-id"]
            assert rid and (not request_id or rid == request_id)
            events = frames(response.text)
            assert events[-1][0] == "complete" and events[-1][1]["status"] == "ok"
            assert all(data["request_id"] == rid for _, data in events)
            assert events[-1][1]["citations"][0]["arxiv_version"] == 7
            assert (await client.get("/healthz")).json() == {"status": "ok"}
            state = (await client.get("/state")).json()
            assert state["provider"]["inflight"] == state["queue_length"] == 0
            assert state["models"][plane.model]["limiter"]["daily_attempts"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": " "},
        {"question": "alpha", "setting": "D"},
        {"question": "alpha", "deadline_ms": 0},
    ],
)
async def test_invalid_requests_never_dispatch(tmp_path: Path, payload: dict[str, object]) -> None:
    async with controlled(tmp_path) as plane:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post("/ask", json=payload)
            assert response.status_code == 422
            assert plane.ledger.daily_usage(plane.model, real=False).attempts == 0


async def test_rejections_are_json_and_unknown_reset_is_omitted(tmp_path: Path) -> None:
    limits = policy().limits["openai/gpt-oss-20b"].model_dump() | {"tpd": 1}
    async with controlled(
        tmp_path, changes={"limits": {"openai/gpt-oss-20b": limits}, "daily_reset_known": False}
    ) as plane:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post("/ask", json={"question": "alpha", "setting": "A"})
            assert (
                response.status_code == 429
                and response.json()["reason"] == "daily_budget_exhausted"
            )
            assert "retry_after_s" not in response.json() and "retry-after" not in response.headers
            assert "midnight" not in response.text and "unverified" in response.json()["message"]
            assert plane.ledger.daily_usage(plane.model, real=False).attempts == 0


async def test_abstention_stream_contains_only_canonical_refusal(tmp_path: Path) -> None:
    async with controlled(tmp_path, reject=True) as plane:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post("/ask", json={"question": "alpha", "setting": "A"})
            assert response.status_code == 200
            events = frames(response.text)
            answer = "".join(data["text"] for event, data in events if event == "answer")
            assert (
                answer == "I couldn't verify a reliable answer from the available paper passages."
            )
            assert "FAKE alpha evidence" not in response.text
            assert events[-1][1]["status"] == "error"


async def test_pressure_preserves_actual_sse_verification(tmp_path: Path) -> None:
    async with controlled(tmp_path) as plane:
        await plane.limiter.update_from_headers(
            plane.model, HeaderSnapshot(observed_ns=0, remaining_tokens_tpm=30000)
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(plane)), base_url="http://127.0.0.1"
        ) as client:
            response = await client.post("/ask", json={"question": "alpha", "setting": "A"})
            complete = frames(response.text)[-1][1]
            assert complete["mode"] == "pressure" and complete["provider_calls"] == 2


async def test_native_http_p3_and_relay_sse_with_real_corpus(
    tmp_path: Path, loopback_ports: set[int]
) -> None:
    if not Path("data/corpus_manifest.json").exists():
        pytest.skip("Pinned corpus requires P1 ingestion")
    async with running(standin_app(profile=profile()), loopback_ports) as url:
        app = create_local_app(
            base_url=url,
            no_cache=True,
            ledger_path=tmp_path / "ledger",
            cache_path=tmp_path / "cache",
            events_path=tmp_path / "events",
        )
        async with app.router.lifespan_context(app):
            async with running(app, loopback_ports) as relay_url:
                async with httpx.AsyncClient(
                    base_url=relay_url.removesuffix("/openai/v1"), trust_env=False, timeout=15
                ) as client:
                    response = await client.post(
                        "/ask",
                        json={"question": "Explain encoder-decoder attention", "setting": "A"},
                        headers={"X-Request-Id": "wire-real-corpus"},
                    )
                    assert response.status_code == 200, response.text
                    events = frames(response.text)
                    complete = events[-1][1]
                    assert complete["status"] == "ok" and complete["provider_calls"] == 2
                    assert complete["citations"][0]["arxiv_id"] == "1706.03762"
                    assert complete["citations"][0]["arxiv_version"] == 7
    with Ledger(tmp_path / "ledger") as ledger:
        usage = ledger.daily_usage(profile().model, real=False)
        assert (
            usage.attempts == 2
            and usage.known_counted_tokens == complete["usage"]["counted_tokens"]
        )
    logs = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
    names = [event["ev"] for event in logs]
    assert names.index("validated") < names.index("first_user_token") < names.index("done")
    assert names.count("arrived") == 1
