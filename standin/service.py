"""Local model-service simulation; same P2 adapter, no credentials or real calls."""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from relay.obs.clock import Clock, SystemClock
from relay.providers.base import LLMRequest
from standin.admin import admin_router
from standin.generation import InvalidInput
from standin.profile import Profile, load_profile
from standin.quota import retry_header
from standin.simulation import Prepared, QuotaRejected, Simulation
from standin.streaming import stream_response

Sleep = Callable[[float], Awaitable[None]]


class ClientGone(ConnectionError):
    """ASGI reports the caller disconnected while service was pending."""


async def delay(
    request: Request, simulation: Simulation, seconds: float, sleep: Sleep, poll_s: float
) -> None:
    until = simulation.clock.monotonic_ns() + int(seconds * 1e9)
    while simulation.clock.monotonic_ns() < until:
        if await request.is_disconnected():
            raise ClientGone("Local client disconnected")
        remaining = (until - simulation.clock.monotonic_ns()) / 1e9
        await sleep(math.nextafter(min(poll_s, remaining), math.inf))


def envelope(prepared: Prepared) -> dict[str, object]:
    return {
        "id": prepared.id,
        "object": "chat.completion",
        "created": prepared.created,
        "model": prepared.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": prepared.text},
                "finish_reason": "stop",
            }
        ],
        "usage": prepared.usage,
    }


def quota_error(exc: QuotaRejected) -> JSONResponse:
    admission = exc.admission
    return JSONResponse(
        status_code=429,
        headers=admission.view.headers | retry_header(admission.retry_after_s),
        content={
            "error": {
                "message": "Stand-in simulated quota exhausted",
                "type": "simulator_quota",
                "code": "simulated_" + admission.scopes[0] + "_exhausted",
                "simulator_only": True,
                "quota_scopes": admission.scopes,
            }
        },
    )


async def nonstream(
    request: Request, prepared: Prepared, simulation: Simulation, sleep: Sleep
) -> JSONResponse:
    success = False
    try:
        fault = prepared.decision.fault
        if fault == "server":
            return JSONResponse(
                status_code=prepared.profile.faults.server_status.value,
                content={"error": {"code": "simulated_server_error"}},
                headers=prepared.admission.view.headers,
            )
        if fault == "timeout":
            await delay(
                request,
                simulation,
                prepared.profile.faults.timeout_s.value,
                sleep,
                prepared.profile.disconnect_poll_s.value,
            )
            return JSONResponse(
                status_code=504,
                content={"error": {"code": "simulated_timeout"}},
                headers=prepared.admission.view.headers,
            )
        total_s = (
            prepared.decision.ttft_ms + prepared.decision.decode_ms(prepared.tokens_out)
        ) / 1000
        await delay(request, simulation, total_s, sleep, prepared.profile.disconnect_poll_s.value)
        success = True
        return JSONResponse(content=envelope(prepared), headers=prepared.admission.view.headers)
    except ClientGone:
        return JSONResponse(status_code=499, content={"error": {"code": "simulated_client_gone"}})
    finally:
        simulation.finish(prepared, success=success)


def create_app(
    *,
    profile: Profile | str = "development",
    clock: Clock | None = None,
    rng: random.Random | None = None,
    sleep: Sleep = asyncio.sleep,
) -> FastAPI:
    selected = load_profile(profile) if isinstance(profile, str) else profile.runnable()
    simulation = Simulation(selected, clock if clock is not None else SystemClock(), rng)
    application = FastAPI(title="Relay local stand-in")
    application.state.simulation = simulation
    application.include_router(admin_router(simulation))

    @application.post("/openai/v1/chat/completions")
    async def completions(payload: LLMRequest, request: Request) -> Response:
        try:
            prepared = simulation.begin(payload)
        except InvalidInput as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "simulated_invalid_input", "message": str(exc)}},
            )
        except QuotaRejected as exc:
            return quota_error(exc)
        if payload.stream and prepared.decision.fault not in {"server", "timeout"}:
            # StreamingResponse owns disconnect monitoring; avoid concurrent ASGI receive readers.
            async def wait(seconds: float) -> None:
                await sleep(seconds)

            return stream_response(prepared, simulation, wait)
        return await nonstream(request, prepared, simulation, sleep)

    return application


app = create_app()
