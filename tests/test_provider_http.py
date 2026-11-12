import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from relay.obs.clock import FakeClock
from relay.obs.events import EventWriter
from relay.providers.base import LLMRequest, Message, ProviderError, ResponseFormat
from relay.providers.groq import GroqCompatibleProvider
from relay.providers.settings import provider_settings
from tests.fake_groq_server import FakeGroqServer, completion
from tests.test_provider_errors import fixture_rules


def request(**changes: object) -> LLMRequest:
    return LLMRequest.model_validate(
        {
            "messages": [Message(role="user", content="FAKE fixture prompt")],
            "max_completion_tokens": 32,
            "response_format": ResponseFormat(type="json_object"),
        }
        | changes
    )


async def test_nonstream_http_usage_headers_and_no_key_required(
    fake_server: FakeGroqServer,
) -> None:
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE-model") as provider:
        result = await provider.complete(request())
        assert provider._client._transport._pool._retries == 0
    assert fake_server.request_count == 1
    assert fake_server.paths == ["/openai/v1/chat/completions"]
    assert result.status_code == 200 and result.usage_complete
    assert result.usage.counted_tokens == 90 and result.usage.reasoning_tokens == 10
    assert result.usage.total_tokens == 120 and result.usage.queue_time == 0.001
    assert result.usage.prompt_time == 0.002 and result.usage.completion_time == 0.003
    assert result.provider_request_id == "FAKE-provider-request"
    assert result.provider_ttft_ms is None and result.first_content_delta_ms is None
    assert "authorization" not in result.headers
    assert result.structured_output == {"answer": "FAKE evidence"}


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("minute", "rate_limit_minute"),
        ("day", "rate_limit_day"),
        ("unknown", "rate_limit_unknown"),
        ("server", "server"),
        ("unavailable", "server"),
        ("auth", "auth"),
        ("malformed", "invalid_response"),
        ("empty", "invalid_response"),
    ],
)
async def test_typed_failures_are_single_attempts(
    fake_server: FakeGroqServer, mode: str, kind: str
) -> None:
    fake_server.mode = mode
    config = provider_settings().model_copy(update={"error_rules": fixture_rules()})
    async with GroqCompatibleProvider(
        fake_server.base_url, None, "FAKE", settings=config
    ) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request())
    assert raised.value.kind == kind and fake_server.request_count == 1
    if mode in {"minute", "day", "unknown"}:
        assert raised.value.status == 429 and raised.value.retry_after_ms == 2000


async def test_missing_usage_is_unknown_and_status_propagates(fake_server: FakeGroqServer) -> None:
    fake_server.mode = "missing-usage"
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE") as provider:
        result = await provider.complete(request())
        assert not result.usage_complete and result.usage.tokens_in is None
        fake_server.mode = "created"
        result = await provider.complete(request())
        assert result.status_code == 201


async def test_key_echo_redacted_in_all_events_and_diagnostics(
    fake_server: FakeGroqServer, tmp_path: Path
) -> None:
    key = "FAKE-key-canary-for-redaction-only"
    fake_server.mode, fake_server.echo = "echo", key + " " + "x" * 1000
    writer = EventWriter(tmp_path / "events.jsonl")
    async with GroqCompatibleProvider(fake_server.base_url, key, "FAKE", events=writer) as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(request())
    raw = writer.path.read_text()
    assert key not in raw and "authorization" not in raw.casefold()
    assert key not in raised.value.body_trunc and len(raised.value.body_trunc) <= 500


async def test_nonstream_duration_uses_injected_clock() -> None:
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))

    def serve(req: httpx.Request) -> httpx.Response:
        clock.advance_ms(12.5)
        clock.set_utc(datetime(2020, 1, 1, tzinfo=UTC))
        return httpx.Response(200, json=completion())

    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        "FAKE",
        clock=clock,
        transport=httpx.MockTransport(serve),
    ) as provider:
        result = await provider.complete(request())
    assert result.total_ms == 12.5 and result.provider_ttft_ms is None
    assert json.loads(result.text)["answer"] == "FAKE evidence"


async def test_request_identifier_falls_back_to_observed_header() -> None:
    body = completion()
    body.pop("x_groq")
    async with GroqCompatibleProvider(
        "http://127.0.0.1/openai/v1",
        None,
        "FAKE",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json=body, headers={"x-request-id": "FAKE-header-request"}
            )
        ),
    ) as provider:
        result = await provider.complete(request())
    assert result.provider_request_id == "FAKE-header-request"


async def test_arbitrary_400_does_not_downgrade_json_schema(fake_server: FakeGroqServer) -> None:
    from relay.providers.base import JsonSchemaDefinition
    from tests.test_provider_contracts import Answer

    fake_server.mode = "bad-request"
    schema = ResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(name="answer", schema_spec=Answer.model_json_schema()),
    )
    async with GroqCompatibleProvider(fake_server.base_url, None, "FAKE") as provider:
        with pytest.raises(ProviderError) as raised:
            await provider.complete(
                request(response_format=schema, schema_stream_policy="json_object")
            )
    assert raised.value.kind == "bad_request" and fake_server.request_count == 1
    assert fake_server.payloads[0]["response_format"]["type"] == "json_schema"


async def test_provider_event_fields_match_architecture_usage_and_headers(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    async with GroqCompatibleProvider(
        fake_server.base_url, None, "FAKE", events=writer
    ) as provider:
        result = await provider.complete(request())
    end = next(
        json.loads(line)
        for line in writer.path.read_text().splitlines()
        if json.loads(line)["ev"] == "call_end"
    )
    assert (end["usage"]["in"], end["usage"]["out"], end["usage"]["cached"]) == (100, 20, 30)
    assert end["hdr"]["rem_req"] == 999 and end["hdr"]["rem_tok"] == 7000
    assert end["quota"]["observed_ns"] == result.header_snapshot.observed_ns
