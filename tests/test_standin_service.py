from pathlib import Path

import httpx
import pytest

from relay.control.ledger import Ledger
from relay.providers.base import LLMRequest, Message, ProviderError, ResponseFormat
from relay.providers.groq import GroqCompatibleProvider
from standin.service import create_app
from tests.standin_helpers import clock, profile


def request(stream: bool = False) -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="Smoke test")],
        stream=stream,
        max_completion_tokens=128,
        response_format=ResponseFormat(type="json_object"),
    )


async def test_nonstream_provider_compatibility_and_local_ledger(tmp_path: Path) -> None:
    c = clock()

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=profile(), clock=c, sleep=sleep)
    with Ledger(tmp_path / "ledger.sqlite3", clock=c) as ledger:
        async with GroqCompatibleProvider(
            "http://127.0.0.1:8081/openai/v1",
            None,
            profile().model,
            clock=c,
            ledger=ledger,
            transport=httpx.ASGITransport(app=app),
        ) as provider:
            result = await provider.complete(request())
        assert result.structured_output == {"answer": "ok"}
        assert (
            result.usage_complete
            and result.usage.total_tokens == result.usage.tokens_in + result.usage.tokens_out
        )
        assert result.usage.reasoning_tokens <= result.usage.tokens_out
        assert result.provider_ttft_ms is None
        assert ledger.session_usage("stand-in", real=False).attempts == 1
        assert ledger.session_usage("stand-in", real=True).attempts == 0
    state = app.state.simulation.snapshot()
    assert state["inflight"] == 0 and state["statistics"]["completed"] == 1


async def test_simulator_429_stays_unknown_to_real_classifier() -> None:
    c = clock()
    p = profile()
    data = p.model_dump()
    data["models"][p.model]["rpm"]["value"] = 1
    from standin.profile import Profile

    p = Profile.model_validate(data)

    async def sleep(seconds: float) -> None:
        c.advance_ms(seconds * 1000)

    app = create_app(profile=p, clock=c, sleep=sleep)
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1", None, p.model, clock=c, transport=httpx.ASGITransport(app=app)
    ) as provider:
        await provider.complete(request())
        with pytest.raises(ProviderError) as caught:
            await provider.complete(request())
    assert caught.value.kind == "rate_limit_unknown"
    assert caught.value.retry_after_ms > 0
    assert caught.value.headers["x-ratelimit-limit-requests"] == str(p.models[p.model].rpd.value)
    assert caught.value.headers["x-ratelimit-limit-tokens"] == str(p.models[p.model].tpm.value)


@pytest.mark.parametrize("status", [500, 503])
async def test_server_fault_is_configured_and_releases_inflight(status: int) -> None:
    app = create_app(profile=profile(faults__server_error_prob=1, faults__server_status=status))
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1", None, profile().model, transport=httpx.ASGITransport(app=app)
    ) as provider:
        with pytest.raises(ProviderError) as caught:
            await provider.complete(request())
    assert caught.value.status == status and caught.value.kind == "server"
    assert app.state.simulation.inflight == 0
