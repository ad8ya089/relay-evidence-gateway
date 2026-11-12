"""Defensive usage extraction and application-side structured-output validation."""

import math

from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from relay.providers.base import LLMRequest, ProviderDiagnostics, Usage


class ResponseState(BaseModel):
    usage: Usage = Field(default_factory=Usage)
    text: str = ""
    completion_id: str | None = None
    response_model: str | None = None
    provider_request_id: str | None = None
    first_reasoning_ms: float | None = None
    first_content_ms: float | None = None
    done: bool = False


def parse_usage(body: dict[str, object], diagnostics: ProviderDiagnostics) -> Usage:
    metadata = body.get("x_groq")
    nested = metadata.get("usage") if isinstance(metadata, dict) else None
    primary = body.get("usage")
    if isinstance(primary, dict):
        data, location = primary, "usage"
    elif isinstance(nested, dict):
        data, location = nested, "x_groq.usage"
    else:
        return Usage()
    if location not in diagnostics.usage_locations:
        diagnostics.usage_locations.append(location)

    def integer(name: str, container: dict[str, object] = data) -> int | None:
        value = container.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if value is not None:
            diagnostics.warnings.append(f"Invalid reported usage field: {name}")
        return None

    def timing(name: str) -> float | None:
        value = data.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value) and value >= 0:
                return float(value)
        if value is not None:
            diagnostics.warnings.append(f"Invalid reported timing field: {name}")
        return None

    prompt = data.get("prompt_tokens_details")
    completion = data.get("completion_tokens_details")
    usage = Usage(
        tokens_in=integer("prompt_tokens"),
        tokens_out=integer("completion_tokens"),
        total_tokens=integer("total_tokens"),
        tokens_cached=integer("cached_tokens", prompt) if isinstance(prompt, dict) else None,
        reasoning_tokens=integer("reasoning_tokens", completion)
        if isinstance(completion, dict)
        else None,
        queue_time=timing("queue_time"),
        prompt_time=timing("prompt_time"),
        completion_time=timing("completion_time"),
    )
    if usage.tokens_cached is not None and usage.tokens_in is not None:
        if usage.tokens_cached > usage.tokens_in:
            usage.tokens_cached = None
            diagnostics.warnings.append(
                "Cached tokens exceed reported input; accounting incomplete"
            )
    if usage.reasoning_tokens is not None and usage.tokens_out is not None:
        if usage.reasoning_tokens > usage.tokens_out:
            usage.reasoning_tokens = None
            diagnostics.warnings.append("Reasoning tokens exceed completion tokens")
    if (
        usage.total_tokens is not None
        and usage.tokens_in is not None
        and usage.tokens_out is not None
    ):
        if usage.total_tokens != usage.tokens_in + usage.tokens_out:
            diagnostics.warnings.append("Reported total differs from input plus completion tokens")
    return usage


def validate_output(text: str, request: LLMRequest) -> dict[str, JsonValue] | None:
    if request.validation_model is not None:
        validated = request.validation_model.model_validate_json(text)
        return TypeAdapter(dict[str, JsonValue]).validate_python(validated.model_dump(mode="json"))
    if request.response_format is not None and request.response_format.type != "text":
        return TypeAdapter(dict[str, JsonValue]).validate_json(text)
    return None
