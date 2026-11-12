import asyncio

import pytest

from relay.providers.base import ProviderError
from relay.providers.groq import GroqCompatibleProvider
from standin.service import create_app
from tests.standin_helpers import profile
from tests.standin_server import running, until_idle
from tests.test_standin_service import request


async def test_real_local_http_stream_parses_content_ttft_and_usage(
    loopback_ports: set[int],
) -> None:
    p = profile()
    app = create_app(profile=p)
    async with running(app, loopback_ports) as url:
        async with GroqCompatibleProvider(url, None, p.model) as provider:
            result = await provider.complete(request(stream=True))
        await until_idle(app)
    assert result.structured_output == {"answer": "ok"}
    assert 0 < result.provider_ttft_ms < result.total_ms
    assert result.usage_complete and result.status_code == 200


@pytest.mark.parametrize("stream", [False, True])
async def test_hanging_service_times_out_and_observes_disconnect(
    loopback_ports: set[int], stream: bool
) -> None:
    p = profile(faults__timeout_prob=1)
    app = create_app(profile=p)
    async with running(app, loopback_ports) as url:
        async with GroqCompatibleProvider(url, None, p.model) as provider:
            with pytest.raises(ProviderError) as caught:
                await provider.complete(
                    request(stream=stream).model_copy(update={"timeout_ms": 80})
                )
        assert caught.value.kind == "timeout"
        await until_idle(app)
    assert app.state.simulation.stats["failed_or_cancelled"] == 1


async def test_cancelling_active_stream_closes_server_lifetime(loopback_ports: set[int]) -> None:
    p = profile(faults__slow_success_prob=1, faults__slow_factor=100)
    app = create_app(profile=p)
    entered = asyncio.Event()
    async with running(app, loopback_ports) as url:
        async with GroqCompatibleProvider(url, None, p.model) as provider:
            pending = asyncio.create_task(provider.complete(request(stream=True)))

            async def wait() -> None:
                while not app.state.simulation.inflight:
                    await asyncio.sleep(0)
                entered.set()

            await asyncio.wait_for(wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await until_idle(app)
    assert entered.is_set() and app.state.simulation.inflight == 0


async def test_real_local_http_detects_incomplete_sse(loopback_ports: set[int]) -> None:
    p = profile(faults__disconnect_prob=1)
    app = create_app(profile=p)
    async with running(app, loopback_ports) as url:
        async with GroqCompatibleProvider(url, None, p.model) as provider:
            with pytest.raises(ProviderError) as caught:
                await provider.complete(request(stream=True))
        await until_idle(app)
    assert caught.value.kind == "disconnect"
