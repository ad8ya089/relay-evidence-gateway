"""Compatible SSE framing and completion-lifetime cleanup."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from standin.simulation import Prepared, Simulation


def sse(payload: dict[str, object]) -> bytes:
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def chunk(
    prepared: Prepared, choices: list[dict[str, object]], **fields: object
) -> dict[str, object]:
    return {
        "id": prepared.id,
        "object": "chat.completion.chunk",
        "created": prepared.created,
        "model": prepared.model,
        "choices": choices,
        **fields,
    }


def split_content(text: str, count: int) -> list[str]:
    count = min(count, len(text))
    return [text[i * len(text) // count : (i + 1) * len(text) // count] for i in range(count)]


async def content_stream(
    prepared: Prepared, simulation: Simulation, wait: Callable[[float], Awaitable[None]]
) -> AsyncIterator[bytes]:
    success = False
    try:
        await wait(prepared.decision.ttft_ms / 1000)
        parts = split_content(prepared.text, prepared.profile.chunks.value)
        truncated_at = max(1, int(len(parts) * prepared.profile.faults.disconnect_fraction.value))
        for index, part in enumerate(parts):
            yield sse(
                chunk(prepared, [{"index": 0, "delta": {"content": part}, "finish_reason": None}])
            )
            await wait(prepared.decision.decode_ms(prepared.tokens_out) / 1000 / len(parts))
            if prepared.decision.fault == "disconnect" and index + 1 >= truncated_at:
                return  # Valid HTTP EOF but intentionally no finish/usage/DONE: incomplete SSE.
        yield sse(chunk(prepared, [{"index": 0, "delta": {}, "finish_reason": "stop"}]))
        if prepared.request.stream_options is None or prepared.request.stream_options.include_usage:
            yield sse(chunk(prepared, [], usage=prepared.usage))
        success = True
        yield b"data: [DONE]\n\n"
    finally:
        simulation.finish(prepared, success=success)


def stream_response(
    prepared: Prepared, simulation: Simulation, wait: Callable[[float], Awaitable[None]]
) -> StreamingResponse:
    return StreamingResponse(
        content_stream(prepared, simulation, wait),
        media_type="text/event-stream",
        headers=prepared.admission.view.headers,
        background=BackgroundTask(simulation.finish, prepared, success=False),
    )
