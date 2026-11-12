import pytest
from pydantic import BaseModel

from relay.providers.base import (
    JsonSchemaDefinition,
    LLMRequest,
    Message,
    ProviderError,
    ProviderFailure,
    ResponseFormat,
    StreamOptions,
    Usage,
)


class Answer(BaseModel):
    answer: str


def test_wire_payload_excludes_local_controls_and_metadata() -> None:
    request = LLMRequest(
        messages=[Message(role="user", content="FAKE prompt")],
        max_completion_tokens=20,
        stream=True,
        response_format=ResponseFormat(type="json_object"),
        validation_model=Answer,
        token_reservation=100,
        capture_response=True,
    )
    payload = request.to_payload("FAKE-model")
    assert payload["max_completion_tokens"] == 20 and "max_tokens" not in payload
    assert payload["stream_options"] == {"include_usage": True}
    assert (
        not {"metadata", "token_reservation", "validation_model", "capture_response"}
        & payload.keys()
    )


def test_schema_contract_does_not_claim_streaming_capability() -> None:
    request = LLMRequest(
        messages=[Message(role="user", content="FAKE prompt")],
        max_completion_tokens=20,
        stream=True,
        response_format=ResponseFormat(
            type="json_schema",
            json_schema=JsonSchemaDefinition(name="answer", schema_spec=Answer.model_json_schema()),
        ),
    )
    assert request.schema_stream_policy == "reject"  # Adapter must enforce this before I/O.
    assert (
        request.to_payload("FAKE")["response_format"]["json_schema"]["schema"]["type"] == "object"
    )
    with pytest.raises(ValueError):
        LLMRequest(
            messages=request.messages, max_completion_tokens=20, stream_options=StreamOptions()
        )


def test_usage_is_nullable_and_reasoning_is_not_added_twice() -> None:
    assert Usage().counted_tokens is None and not Usage().complete
    usage = Usage(
        tokens_in=100, tokens_cached=30, tokens_out=20, total_tokens=120, reasoning_tokens=10
    )
    assert usage.counted_tokens == 90 and usage.complete
    assert not usage.model_copy(update={"total_tokens": 130}).complete
    error = ProviderError(ProviderFailure(kind="rate_limit_unknown", status_code=429))
    assert error.kind == "rate_limit_unknown" and error.status == 429
    event_usage = usage.model_dump(by_alias=True)
    assert Usage.model_validate(event_usage).counted_tokens == 90
