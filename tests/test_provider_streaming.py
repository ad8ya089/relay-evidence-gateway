import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from relay.obs.clock import FakeClock
from relay.obs.events import EventWriter
from relay.providers.base import JsonSchemaDefinition, ProviderError, ResponseFormat, StreamDelta
from relay.providers.groq import GroqCompatibleProvider
from relay.providers.settings import provider_settings
from relay.providers.streaming import sse_data
from tests.fake_groq_server import FakeGroqServer, completion
from tests.test_provider_contracts import Answer
from tests.test_provider_http import request


@pytest.mark.parametrize(
    "mode,path",
    [
        ("usage-final", "usage"),
        ("usage-only", "usage"),
        ("x-groq-usage", "x_groq.usage"),
    ],
)
async def test_stream_usage_locations_and_reasoning_content(
    fake_server: FakeGroqServer,
    mode: str,
    path: str,
    tmp_path: Path,
) -> None:
    fake_server.mode = mode
    deltas: list[StreamDelta] = []

    async def collect(delta: StreamDelta) -> None:
        deltas.append(delta)

    writer = EventWriter(tmp_path / "events.jsonl")
    async with GroqCompatibleProvider(
        fake_server.base_url, None, "FAKE", events=writer
    ) as provider:
        result = await provider.complete(
            request(stream=True, validation_model=Answer), on_delta=collect
        )
    assert result.structured_output == {"answer": "FAKE evidence"}
    assert result.usage_complete and result.usage.counted_tokens == 90
    assert result.provider_ttft_ms == result.first_content_delta_ms
    assert result.first_reasoning_delta_ms <= result.first_content_delta_ms
    assert [d.what for d in deltas] == ["reasoning", "content", "content"]
    assert path in result.diagnostics.usage_locations
    assert fake_server.request_count == 1
    events = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [r["what"] for r in events if r["ev"] == "first_delta"] == ["reasoning", "content"]


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("malformed-stream", "invalid_response"),
        ("disconnect", "disconnect"),
        ("incomplete", "disconnect"),
        ("empty-stream", "invalid_response"),
    ],
)
async def test_malformed_incomplete_and_disconnected_streams(
    fake_server: FakeGroqServer,
    mode: str,
    kind: str,
) -> None:
    fake_server.mode = mode
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE") as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request(stream=True))
    assert raised.value.kind == kind and fake_server.request_count == 1


async def test_absent_stream_usage_is_null(fake_server: FakeGroqServer) -> None:
    fake_server.mode = "no-usage"
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE") as provider:
        result = await provider.complete(request(stream=True))
    assert result.usage.tokens_in is None and not result.usage_complete


async def test_timeout_closes_stream_and_server_observes_disconnect(
    fake_server: FakeGroqServer,
) -> None:
    fake_server.mode = "hang"
    config = provider_settings().model_copy(update={"read_timeout_s": 0.05})
    async with GroqCompatibleProvider(
        fake_server.base_url, None, "FAKE", settings=config
    ) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request(stream=True))
        assert raised.value.kind == "timeout"
        await asyncio.wait_for(fake_server.disconnected.wait(), 2)
    assert fake_server.request_count == 1


async def test_caller_cancellation_closes_stream(fake_server: FakeGroqServer) -> None:
    fake_server.mode = "hang"
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE") as provider:
        task = asyncio.create_task(provider.complete(request(stream=True)))
        await asyncio.wait_for(fake_server.entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(fake_server.disconnected.wait(), 2)
    assert fake_server.request_count == 1


async def test_schema_stream_rejected_or_explicitly_converted(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    schema = ResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(name="answer", schema_spec=Answer.model_json_schema()),
    )
    writer = EventWriter(tmp_path / "events.jsonl")
    async with GroqCompatibleProvider(
        fake_server.base_url, None, "FAKE", events=writer
    ) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request(stream=True, response_format=schema))
        assert raised.value.kind == "unsupported_combination" and fake_server.request_count == 0
        result = await provider.complete(
            request(
                stream=True,
                response_format=schema,
                schema_stream_policy="json_object",
                validation_model=Answer,
            )
        )
    assert result.structured_output and fake_server.request_count == 1
    assert fake_server.payloads[0]["response_format"] == {"type": "json_object"}
    assert "compatibility_fallback" in writer.path.read_text()


class TimedStream(httpx.AsyncByteStream):
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        frames = [
            (5, ": keepalive\n\n"),
            (5, 'data: {"choices":[{"delta":{"reasoning":"FAKE reasoning"}}]}\n\n'),
            (3, 'data: {"choices":[{"delta":{"content":""}}]}\n\n'),
            (7, 'data: {"choices":[{"delta":{"content":"{\\"answer\\":"}}]}\n\n'),
            (5, 'data: {"choices":[{"delta":{"content":"\\"FAKE\\"}"}}]}\n\n'),
            (5, "data: " + json.dumps({"choices": [], "usage": completion()["usage"]}) + "\n\n"),
            (5, "data: [DONE]\n\n"),
        ]
        for milliseconds, frame in frames:
            self.clock.advance_ms(milliseconds)
            yield frame.encode()

    async def aclose(self) -> None:
        self.closed = True


async def test_ttft_is_first_nonempty_content_with_deterministic_clock() -> None:
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))
    stream = TimedStream(clock)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    )
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1", None, "FAKE", clock=clock, transport=transport
    ) as provider:
        result = await provider.complete(request(stream=True))
    assert result.first_reasoning_delta_ms == 10
    assert result.first_content_delta_ms == result.provider_ttft_ms == 20
    assert result.total_ms == 35 and stream.closed


async def test_multiline_data_comments_and_usage_only_framing() -> None:
    async def lines() -> AsyncIterator[str]:
        for line in (": comment", "", "data: {", 'data: "choices":[]', "data: }", "", "data:", ""):
            yield line

    values = [value async for value in sse_data(lines())]
    assert json.loads(values[0]) == {"choices": []}
    assert values[1] == ""
