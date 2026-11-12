"""SSE framing and deltas, independent of the surrounding provider lifecycle."""

import json
from collections.abc import AsyncIterator

import httpx

from relay.obs.clock import Clock
from relay.obs.events import Sanitizer
from relay.providers.base import DeltaCallback, LLMRequest, ProviderDiagnostics, StreamDelta
from relay.providers.parsing import ResponseState, parse_usage


class StreamDisconnectedError(ConnectionError):
    """EOF before the explicit DONE terminator."""


class DeltaCallbackError(RuntimeError):
    """An application callback failed; do not repeat the request."""


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        # Comments, id/event/retry fields and empty keep-alives carry no deltas.
    if data:
        yield "\n".join(data)


async def consume_stream(
    response: httpx.Response,
    request: LLMRequest,
    state: ResponseState,
    diagnostics: ProviderDiagnostics,
    *,
    clock: Clock,
    start_ns: int,
    sanitizer: Sanitizer,
    on_delta: DeltaCallback,
) -> None:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        raise ValueError("Expected an SSE response")
    async for data in sse_data(response.aiter_lines()):
        if not data.strip():
            continue
        if data.strip() == "[DONE]":
            state.done = True
            break
        diagnostics.body_trunc = sanitizer.text(data, body=True)
        if request.capture_response:
            diagnostics.stream_chunks.append(diagnostics.body_trunc)
        body = json.loads(data)
        if not isinstance(body, dict) or "error" in body:
            raise ValueError("Invalid SSE completion envelope")
        diagnostics.response_fields = sorted(
            set(diagnostics.response_fields).union(sanitizer.text(key) for key in body)
        )
        reported = parse_usage(body, diagnostics)
        # Usage chunks are snapshots, never additive attempts/token quantities.
        state.usage = state.usage.model_copy(update=reported.model_dump(exclude_none=True))
        for field, wire in (("completion_id", "id"), ("response_model", "model")):
            if isinstance(value := body.get(wire), str):
                setattr(state, field, sanitizer.text(value))
        metadata = body.get("x_groq")
        if isinstance(metadata, dict) and isinstance(metadata.get("id"), str):
            state.provider_request_id = sanitizer.text(metadata["id"])
        choices = body.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("Malformed SSE choices")
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                raise ValueError("Malformed SSE delta")
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning", delta.get("reasoning_content"))
            for what, value in (("reasoning", reasoning), ("content", delta.get("content"))):
                if value is None or value == "":
                    continue
                if not isinstance(value, str):
                    raise ValueError("Delta text must be a string")
                elapsed = (clock.monotonic_ns() - start_ns) / 1_000_000
                if what == "reasoning" and state.first_reasoning_ms is None:
                    state.first_reasoning_ms = elapsed
                if what == "content":
                    if state.first_content_ms is None:
                        state.first_content_ms = elapsed
                    state.text += value
                await on_delta(StreamDelta(what=what, text=value, t_ms=elapsed))
    if not state.done:
        raise StreamDisconnectedError("Stream ended before DONE")
    if not state.text:
        raise ValueError("Stream completed without content")
    if not request.capture_response:
        diagnostics.body_trunc = None
