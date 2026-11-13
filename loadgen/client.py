"""Actual HTTP open-loop dispatch; no provider calls, retries or answer logging."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from loadgen.arrivals import PlannedRequest
from loadgen.models import ClientRecord, RunContext
from loadgen.settings import LoadSettings
from loadgen.sse import ProtocolError, SSEDecoder
from relay.app.models import Citation
from relay.obs.clock import Clock
from relay.obs.events import EventContext, EventWriter
from relay.providers.groq import is_loopback_url

Sleep = Callable[[float], Awaitable[None]]


class ObservedUsage(BaseModel):
    tokens_in: int | None = Field(ge=0)
    tokens_out: int | None = Field(ge=0)
    tokens_cached: int | None = Field(ge=0)
    counted_tokens: int | None = Field(ge=0)
    complete: bool

    @model_validator(mode="after")
    def consistent(self) -> "ObservedUsage":
        if self.complete and any(
            value is None
            for value in (self.tokens_in, self.tokens_out, self.tokens_cached, self.counted_tokens)
        ):
            raise ValueError("Complete usage cannot be unknown")
        if (
            self.tokens_in is not None
            and self.tokens_cached is not None
            and self.tokens_cached > self.tokens_in
        ):
            raise ValueError("Cached input is not additional input")
        if (
            all(
                v is not None
                for v in (self.tokens_in, self.tokens_out, self.tokens_cached, self.counted_tokens)
            )
            and self.counted_tokens != self.tokens_in - self.tokens_cached + self.tokens_out
        ):
            raise ValueError("Counted usage contradicts cached-input accounting")
        return self


class Completion(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    status: str
    provider_calls: int = Field(ge=0)
    retries: int = Field(ge=0)
    usage: ObservedUsage
    mode: str
    cache_hit: bool
    coalesced: bool
    citations: list[Citation]


class LoadClient:
    def __init__(
        self,
        url: str,
        config: LoadSettings,
        clock: Clock,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not is_loopback_url(url):
            raise ValueError("Load clients require an explicit local Relay URL")
        self.clock, self.config = clock, config
        self.http = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/",
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(
                connect=config.connect_timeout_s,
                pool=config.pool_timeout_s,
                read=None,
                write=config.connect_timeout_s,
            ),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
            ),
            transport=transport
            or httpx.AsyncHTTPTransport(
                retries=0,
                limits=httpx.Limits(
                    max_connections=config.max_connections,
                    max_keepalive_connections=config.max_keepalive_connections,
                ),
            ),
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def measure(
        self,
        item: PlannedRequest,
        question: str,
        context: RunContext,
        scheduled_ns: int,
        scheduled_utc: str,
        writers: list[EventWriter],
    ) -> ClientRecord:
        clock = self.clock
        task_ns = clock.monotonic_ns()
        task_utc = clock.utc_now().astimezone(UTC).isoformat(timespec="milliseconds")
        actual_ns: int | None = None
        actual_utc: str | None = None
        first: int | None = None
        complete = False
        metrics: dict[str, object] = {}
        status, failure = "error", "incomplete_stream"
        http_status = None
        cancelled = False
        scope = asyncio.timeout(None)
        deadline_ns = task_ns + int(self.config.pool_timeout_s * 1e9)
        changed = asyncio.Event()
        timer: asyncio.Task[None] | None = None

        def set_deadline(value: int) -> None:
            nonlocal deadline_ns
            deadline_ns = value
            changed.set()

        async def expire() -> None:
            # Windows event-loop timers may wake early. Only this clock decides expiry;
            # early wakeups cancel an Event wait, never the in-flight HTTP stream.
            while True:
                remaining = (deadline_ns - clock.monotonic_ns()) / 1e9
                if remaining <= 0:
                    scope.reschedule(asyncio.get_running_loop().time())
                    return
                changed.clear()
                try:
                    await asyncio.wait_for(changed.wait(), remaining)
                except TimeoutError:
                    pass

        async def trace(name: str, info: dict[str, object]) -> None:
            nonlocal actual_ns, actual_utc
            if name.endswith("send_request_headers.started") and actual_ns is None:
                actual_ns = clock.monotonic_ns()
                actual_utc = clock.utc_now().astimezone(UTC).isoformat(timespec="milliseconds")
                set_deadline(actual_ns + context.deadline_ms * 1_000_000)

        try:
            async with scope:
                timer = asyncio.create_task(expire())
                async with self.http.stream(
                    "POST",
                    "ask",
                    json={
                        "question": question,
                        "setting": item.setting,
                        "deadline_ms": context.deadline_ms,
                        "cache": context.cache,
                    },
                    headers={"X-Request-Id": item.request_id, "Accept-Encoding": "identity"},
                    extensions={"trace": trace},
                ) as response:
                    http_status = response.status_code
                    if actual_ns is None:
                        # Explicitly marked estimate for non-network injected test transports.
                        actual_ns = task_ns
                        set_deadline(actual_ns + context.deadline_ms * 1_000_000)
                    if response.status_code != 200:
                        status = "rate_limited" if response.status_code == 429 else "error"
                        failure = f"http_{response.status_code}"
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > self.config.max_sse_frame_bytes:
                                break
                        try:
                            reason = json.loads(body).get("reason")
                            if (
                                isinstance(reason, str)
                                and len(reason) <= 500
                                and re.fullmatch(r"[a-z0-9_]+", reason)
                            ):
                                failure = reason
                        except (ValueError, AttributeError):
                            pass
                        del body
                    else:
                        if "text/event-stream" not in response.headers.get("content-type", ""):
                            raise ProtocolError("wrong_sse_content_type")
                        decoder = SSEDecoder(self.config.max_sse_frame_bytes)
                        async for chunk in response.aiter_bytes():
                            for frame in decoder.feed(chunk, clock.monotonic_ns()):
                                if (
                                    frame.event in {"answer", "complete", "error"}
                                    and frame.data.get("request_id") != item.request_id
                                ):
                                    raise ProtocolError("sse_request_id_mismatch")
                                if complete:
                                    raise ProtocolError("sse_after_completion")
                                if frame.event == "answer":
                                    if not isinstance(frame.data.get("text"), str):
                                        raise ProtocolError("invalid_answer_fragment")
                                    if frame.data["text"] and first is None:
                                        first = frame.answer_byte_ns
                                elif frame.event == "error":
                                    raise ProtocolError("server_sse_error")
                                elif frame.event == "complete":
                                    if first is None or frame.data.get("status") not in {
                                        "ok",
                                        "error",
                                        "timeout",
                                        "rate_limited",
                                    }:
                                        raise ProtocolError("invalid_completion")
                                    try:
                                        parsed = Completion.model_validate(frame.data)
                                    except ValueError:
                                        raise ProtocolError("invalid_completion_metadata") from None
                                    metrics = parsed.model_dump(
                                        include={
                                            "provider_calls",
                                            "retries",
                                            "mode",
                                            "cache_hit",
                                            "coalesced",
                                        }
                                    )
                                    metrics.update(parsed.usage.model_dump(exclude={"complete"}))
                                    metrics["usage_complete"] = parsed.usage.complete
                                    metrics["citation_valid"] = True if parsed.citations else None
                                    status = frame.data["status"]
                                    failure = None if status == "ok" else "validated_abstention"
                                    complete = True
                            if complete:
                                break
                        if not complete:
                            decoder.finish()
                        if not complete:
                            raise ProtocolError("incomplete_stream")
        except TimeoutError:
            status, failure = (
                "timeout",
                "client_gave_up" if actual_ns is not None else "connection_or_pool_timeout",
            )
        except asyncio.CancelledError:
            status, failure, cancelled = "timeout", "client_cancelled", True
        except ProtocolError as exc:
            status, failure = "error", str(exc)
        except httpx.TransportError as exc:
            status, failure = "error", "transport_" + type(exc).__name__
        except Exception as exc:
            status, failure = "error", "client_" + type(exc).__name__
        finally:
            if timer:
                if not timer.done():
                    timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)
        finished = clock.monotonic_ns()
        wire_observed = actual_utc is not None
        actual_ns = actual_ns if actual_ns is not None else task_ns
        actual_utc = actual_utc or task_utc
        row = ClientRecord(
            request_id=item.request_id,
            run_id=context.run_id,
            scenario=context.scenario,
            source=context.source,
            model=context.model,
            question_id=item.question_id,
            seed=item.seed,
            sampling=item.sampling,
            sample_draw=item.sample_draw,
            fixture_only=context.fixture_only,
            scheduled_arrival_monotonic=scheduled_ns,
            task_dispatch_monotonic=task_ns,
            actual_send_monotonic=actual_ns,
            first_answer_byte_monotonic=first,
            completion_monotonic=finished,
            scheduled_arrival_utc=scheduled_utc,
            actual_send_utc=actual_utc,
            dispatch_timestamp_source="httpcore_headers"
            if wire_observed
            else "transport_entry_estimate",
            generator_lag_ms=max(0, (actual_ns - scheduled_ns) / 1e6),
            transport_wait_ms=max(0, (actual_ns - task_ns) / 1e6),
            first_token_ms=(first - actual_ns) / 1e6
            if first is not None and status == "ok"
            else None,
            total_ms=(finished - actual_ns) / 1e6,
            status=status,
            failure_reason=failure,
            http_status_last=http_status,
            deadline_ms=context.deadline_ms,
            retrieval_setting=item.setting,
            terminal_complete=complete,
            **metrics,
        )
        for writer in writers:
            writer.write(
                EventContext(rid=item.request_id, run=context.run_id, arrival_ns=task_ns),
                "client_result",
                **row.model_dump(),
            )
        if cancelled:
            raise asyncio.CancelledError
        return row


async def dispatch(
    client: LoadClient,
    plans: list[PlannedRequest],
    questions: dict[str, str],
    context: RunContext,
    writers: list[EventWriter],
    *,
    sleep: Sleep = asyncio.sleep,
    start_ns: int | None = None,
    start_utc: datetime | None = None,
) -> list[ClientRecord]:
    if len(plans) > client.config.max_requests:
        raise ValueError("Bounded task count exceeded")
    start = client.clock.monotonic_ns() if start_ns is None else start_ns
    utc = (start_utc or client.clock.utc_now()).astimezone(UTC)
    tasks = []
    try:
        for item in plans:
            scheduled = start + item.offset_ns
            remaining = (scheduled - client.clock.monotonic_ns()) / 1e9
            while remaining > 0:
                await sleep(remaining)
                remaining = (scheduled - client.clock.monotonic_ns()) / 1e9
            stamp = (utc + timedelta(microseconds=item.offset_ns / 1000)).isoformat(
                timespec="milliseconds"
            )
            tasks.append(
                asyncio.create_task(
                    client.measure(
                        item, questions[item.question_id], context, scheduled, stamp, writers
                    )
                )
            )
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
