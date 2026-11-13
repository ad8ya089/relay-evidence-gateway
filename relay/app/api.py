"""Local, single-worker HTTP boundary; only P4-approved answers enter SSE."""

import argparse
import asyncio
import ipaddress
import json
import math
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, Field, field_validator
from starlette.responses import Response

from relay.app.cache import AnswerCache
from relay.app.delivery import DeliveryError, approved, iter_validated_answer
from relay.app.messages import message
from relay.app.models import Question, StrictModel
from relay.app.singleflight import SingleFlight
from relay.control.admission import Admission, Reject
from relay.control.breaker import CircuitBreaker
from relay.control.ledger import Ledger
from relay.control.limiter import Limiter
from relay.control.plane import ControlPlane
from relay.control.retry import RetryBudget
from relay.control.scheduler import ProviderScheduler
from relay.control.settings import load_policy
from relay.obs.clock import SystemClock
from relay.obs.events import EventContext, EventWriteError, EventWriter
from relay.providers.groq import GroqCompatibleProvider, is_loopback_url
from relay.providers.settings import provider_settings
from relay.retrieval.models import CorpusManifest
from relay.retrieval.search import RetrievalEngine
from relay.settings import PaperId
from relay.tools.executor import ToolExecutor
from standin.profile import load_profile


class AskPayload(StrictModel):
    question: str = Field(validation_alias=AliasChoices("question", "text"))
    setting: str = "C"
    deadline_ms: int | None = Field(default=None, gt=0)
    paper_scope: PaperId | None = None
    cache: bool | None = None

    @field_validator("question")
    @classmethod
    def valid_question(cls, value: str) -> str:
        return Question(text=value).text

    @field_validator("setting")
    @classmethod
    def online_setting(cls, value: str) -> str:
        if value not in {"A", "B", "C"}:
            raise ValueError("Only online retrieval settings are accepted")
        return value


def rejection_response(rejection: Reject, request_id: str) -> JSONResponse:
    content: dict[str, object] = {"message": message(rejection.reason), "reason": rejection.reason}
    headers = {"X-Request-Id": request_id}
    if rejection.retry_after_s is not None:
        content["retry_after_s"] = rejection.retry_after_s
        headers["Retry-After"] = str(math.ceil(rejection.retry_after_s))
    return JSONResponse(content=content, status_code=rejection.status_code, headers=headers)


def sse(event: str, data: dict[str, object]) -> str:
    return (
        "event: "
        + event
        + "\ndata: "
        + json.dumps(data, ensure_ascii=True, allow_nan=False)
        + "\n\n"
    )


def create_app(
    control: ControlPlane | None = None,
    *,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    if control is None and lifespan is None:
        raise ValueError("Explicit dependencies or a configured lifespan are required")
    app = FastAPI(lifespan=lifespan)
    if control is not None:
        app.state.control = control

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {"message": "Invalid question payload.", "reason": "invalid_request"}, status_code=422
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state")
    async def state(request: Request) -> Response:
        try:
            local = (
                request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
            )
        except ValueError:
            local = False
        if not local:
            return JSONResponse({"message": "Local endpoint."}, status_code=403)
        plane: ControlPlane = request.app.state.control
        return JSONResponse(await plane.state())

    @app.post("/ask")
    async def ask(
        payload: AskPayload, request: Request, x_request_id: str | None = Header(default=None)
    ) -> Response:
        plane: ControlPlane = request.app.state.control
        rid = x_request_id or uuid4().hex
        if (
            len(rid) > plane.policy.server.request_id_max_chars
            or re.fullmatch(r"[A-Za-z0-9._:-]+", rid) is None
        ):
            return JSONResponse(
                {"message": "Invalid request identifier.", "reason": "invalid_request"},
                status_code=422,
            )
        question = Question(
            text=payload.question,
            setting=payload.setting,
            request_id=rid,
            deadline_ms=payload.deadline_ms,
            paper_scope=payload.paper_scope,
        )

        async def disconnect() -> None:
            while not await request.is_disconnected():
                await asyncio.sleep(plane.policy.server.disconnect_poll_ms / 1000)

        arrival = plane.clock.monotonic_ns()

        def client_gone() -> None:
            plane.events.write(
                EventContext(rid=rid, run="p5-local", arrival_ns=arrival),
                "client_disconnected",
                phase="preparing",
            )

        work = asyncio.create_task(
            plane.ask(question, cache_enabled=payload.cache, arrival_ns=arrival)
        )
        monitor = asyncio.create_task(disconnect())
        try:
            done, _ = await asyncio.wait({work, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if monitor in done and work not in done:
                client_gone()
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                return Response(status_code=499)
            result = await work
        except asyncio.CancelledError:
            client_gone()
            raise
        except Exception:
            return rejection_response(Reject(reason="provider_unavailable", status_code=503), rid)
        finally:
            monitor.cancel()
            if not work.done():
                work.cancel()
            await asyncio.gather(work, monitor, return_exceptions=True)
        if isinstance(result, Reject):
            return rejection_response(result, rid)
        if not approved(result) or plane.clock.monotonic_ns() >= result.deadline_ns:
            reason = {
                "provider_rate_limit_minute": "provider_429_minute",
                "provider_rate_limit_day": "provider_429_day",
                "provider_rate_limit_unknown": "provider_429_unknown",
                "deadline": "deadline_infeasible",
            }.get(
                result.failure_reason or "",
                result.failure_reason
                if result.failure_reason
                in {
                    "breaker_open",
                    "daily_budget_exhausted",
                    "projected_wait",
                    "deadline_infeasible",
                }
                else "provider_unavailable",
            )
            quota_view = await plane.limiter.remaining(plane.model)
            return rejection_response(
                Reject(
                    reason=reason,
                    retry_after_s=quota_view.retry_after_s
                    if quota_view.blocked == reason
                    else None,
                    status_code=429
                    if reason.startswith("provider_429")
                    or reason in {"daily_budget_exhausted", "projected_wait"}
                    else 503,
                ),
                rid,
            )

        async def stream() -> AsyncIterator[str]:
            try:
                async for piece in iter_validated_answer(
                    result, clock=plane.clock, events=plane.events
                ):
                    yield sse("answer", {"request_id": rid, "text": piece})
                yield sse(
                    "complete",
                    {
                        "request_id": rid,
                        "status": result.status,
                        "cache_hit": result.cache_hit,
                        "coalesced": result.coalesced,
                        "mode": result.mode,
                        "citations": [c.model_dump() for c in result.citations],
                        "provider_calls": result.provider_calls,
                        "retries": result.retries,
                        "usage": result.usage.model_dump(),
                        "first_token_ms": result.first_token_ms,
                    },
                )
            except (DeliveryError, EventWriteError):
                yield sse("error", {"request_id": rid, "message": message("deadline_infeasible")})
            except asyncio.CancelledError:
                plane.events.write(
                    EventContext(rid=rid, run=result.run_id, arrival_ns=result.arrival_ns),
                    "client_disconnected",
                    state=result.state.value,
                )
                raise

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Request-Id": rid, "Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


def create_local_app(
    *,
    policy_path: Path = Path("config/policy.yaml"),
    base_url: str | None = None,
    no_cache: bool = False,
    ledger_path: Path = Path("results/relay.sqlite3"),
    cache_path: Path = Path("results/relay_cache.sqlite3"),
    events_path: Path = Path("results/events.jsonl"),
    allow_unsafe_naive: bool = False,
) -> FastAPI:
    policy = load_policy(policy_path)
    if no_cache:
        policy = type(policy).model_validate(
            policy.model_dump()
            | {"cache": policy.cache.model_dump() | {"enabled": False, "singleflight": False}}
        )
    url = base_url or policy.server.standin_base_url
    if not is_loopback_url(url):
        raise ValueError("P5 local service accepts loopback stand-ins only")
    if policy.retry.policy == "naive_s7a" and not allow_unsafe_naive:
        raise ValueError("Unsafe naive policy requires --allow-unsafe-naive")
    manifest = CorpusManifest.model_validate_json(
        Path("data/corpus_manifest.json").read_text(encoding="utf-8")
    )
    model = load_profile("development").model

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        clock = SystemClock()
        events = EventWriter(events_path, clock=clock)
        cache = AnswerCache(cache_path, policy.cache, clock)
        flight = SingleFlight(policy.cache, clock)
        with Ledger(ledger_path, clock=clock) as ledger:
            async with GroqCompatibleProvider(
                url,
                None,
                model,
                ledger=ledger,
                events=events,
                clock=clock,
                settings=provider_settings().model_copy(
                    update={"error_rules": policy.standin_error_rules}
                ),
            ) as provider:
                limiter = Limiter(policy, ledger, clock)
                breaker = CircuitBreaker(policy.breaker, clock)
                engine = RetrievalEngine(clock=clock)
                app.state.control = ControlPlane(
                    policy=policy,
                    clock=clock,
                    provider=provider,
                    ledger=ledger,
                    events=events,
                    executor=ToolExecutor(engine.search_papers, clock=clock, events=events),
                    model=model,
                    corpus_hash=manifest.corpus_hash,
                    cache=cache,
                    singleflight=flight,
                    limiter=limiter,
                    admission=Admission(policy, limiter, breaker, clock),
                    scheduler=ProviderScheduler(
                        policy.max_inflight_provider, policy.scheduler, clock
                    ),
                    retry_budget=RetryBudget(policy.retry),
                    breaker=breaker,
                    allow_unsafe_naive=allow_unsafe_naive,
                )
                try:
                    yield
                finally:
                    await flight.close()
                    cache.close()

    return create_app(lifespan=lifespan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("config/policy.yaml"))
    parser.add_argument("--base-url")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-unsafe-naive", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        import uvicorn

        policy = load_policy(args.policy)
        app = create_local_app(
            policy_path=args.policy,
            base_url=args.base_url,
            no_cache=args.no_cache,
            allow_unsafe_naive=args.allow_unsafe_naive,
        )
        port = policy.server.port if args.port is None else args.port
        if not 0 <= port <= 65535:
            raise ValueError("Invalid local port")
        if args.check:
            print("Relay startup valid: stand-in, one worker, local-only")
            return 0
        uvicorn.run(
            app,
            host=policy.server.host,
            port=port,
            workers=1,
            proxy_headers=False,
            timeout_graceful_shutdown=policy.server.graceful_shutdown_s,
        )
        return 0
    except (ValueError, OSError):
        print("Relay refused: invalid local configuration or unavailable corpus")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
