"""One raw-httpx adapter for loopback stand-ins and guarded external providers."""

import asyncio
import ipaddress
import json
from contextlib import nullcontext

import httpx
from pydantic import SecretStr

from relay.control.ledger import CallRecord, Ledger, LedgerError
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriteError, EventWriter, Sanitizer
from relay.providers.base import (
    DeltaCallback,
    ErrorKind,
    HeaderSnapshot,
    LLMRequest,
    LLMResult,
    ProviderCancelledError,
    ProviderDiagnostics,
    ProviderError,
    ProviderFailure,
    ResponseFormat,
    StreamDelta,
    Usage,
)
from relay.providers.errors import classify_error, header_snapshot
from relay.providers.guard import GuardSettlement, RealCallDenied, RealCallGuard
from relay.providers.parsing import ResponseState, parse_usage, validate_output
from relay.providers.settings import ProviderSettings, provider_settings
from relay.providers.streaming import DeltaCallbackError, StreamDisconnectedError, consume_stream


def is_loopback_url(base_url: str) -> bool:
    url = httpx.URL(base_url)
    if (
        url.scheme not in {"http", "https"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Provider URL requires HTTP(S), with no credentials/query/fragment")
    if url.host == "localhost":
        return True
    try:
        return ipaddress.ip_address(url.host).is_loopback
    except ValueError:
        return False


class GroqCompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        settings: ProviderSettings | None = None,
        clock: Clock | None = None,
        events: EventWriter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        ledger: Ledger | None = None,
        guard: RealCallGuard | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings if settings is not None else provider_settings()
        self.clock = clock if clock is not None else SystemClock()
        self.events = events
        self.ledger = ledger
        self.guard = guard
        self.session_id = guard.session_id if guard is not None else session_id or "stand-in"
        if guard is not None and guard.ledger is not ledger:
            raise ValueError("Provider and guard must share the same ledger")
        self.model = model
        self.is_real = not is_loopback_url(base_url)
        self._api_key = SecretStr(api_key or "")
        self.sanitizer = Sanitizer([self._api_key.get_secret_value()])
        if events is not None:
            events.sanitizer.register_secret(self._api_key.get_secret_value())
        url = httpx.URL(base_url)
        if url.host == "localhost":
            url = url.copy_with(host="127.0.0.1")
        self._client = httpx.AsyncClient(
            base_url=str(url).rstrip("/") + "/",
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                connect=self.settings.connect_timeout_s,
                read=self.settings.read_timeout_s,
                write=self.settings.write_timeout_s,
                pool=self.settings.pool_timeout_s,
            ),
            transport=transport if transport is not None else httpx.AsyncHTTPTransport(retries=0),
        )

    async def __aenter__(self) -> "GroqCompatibleProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _emit(self, context: EventContext, ev: str, **fields: object) -> None:
        if self.events is not None:
            self.events.write(context, ev, **fields)

    def _error(
        self,
        kind: ErrorKind,
        start: int,
        snapshot: HeaderSnapshot,
        *,
        body: str = "",
        usage: Usage | None = None,
        reason: str = "",
    ) -> ProviderError:
        return ProviderError(
            ProviderFailure(
                kind=kind,
                headers=snapshot.headers,
                body_trunc=self.sanitizer.text(body, body=True),
                usage=usage or Usage(),
                total_ms=(self.clock.monotonic_ns() - start) / 1_000_000,
                safe_context={"reason": self.sanitizer.text(reason, body=True)},
            )
        )

    def _prepare(self, request: LLMRequest, context: EventContext) -> LLMRequest:
        if self.is_real:
            url = self._client.base_url
            if (
                url.scheme != "https"
                or url.host not in self.settings.allowed_real_hosts
                or url.port not in {None, 443}
            ):
                raise RealCallDenied("External origin is not an allowed TLS provider")
            if self.guard is None or self.ledger is None:
                raise RealCallDenied("External calls require a durable real-call guard and ledger")
            self.guard.check_permissions()
            if not self._api_key.get_secret_value():
                raise RealCallDenied("An API credential is required for this external provider")
            if request.token_reservation is None:
                raise RealCallDenied("A conservative token reservation is required before dispatch")
        if request.stream and request.response_format is not None:
            if request.response_format.type == "json_schema":
                if request.schema_stream_policy == "reject":
                    raise ProviderError(
                        ProviderFailure(
                            kind="unsupported_combination",
                            safe_context={
                                "reason": "Structured Outputs with streaming is unsupported"
                            },
                        )
                    )
                if request.schema_stream_policy == "json_object":
                    self._emit(
                        context,
                        "compatibility_fallback",
                        reason="documented_schema_stream",
                        requested="json_schema",
                        actual="json_object",
                    )
                    request = request.model_copy(
                        update={"response_format": ResponseFormat(type="json_object")}
                    )
        return request

    def _begin_attempt(self, request: LLMRequest) -> str | None:
        if self.ledger is None:
            return None
        safe = self.sanitizer.text
        record = CallRecord(
            id=safe(request.metadata.call_id),
            run_id=safe(request.metadata.run_id),
            ts_utc=self.clock.utc_now(),
            kind=safe(request.metadata.kind),
            model=safe(request.model or self.model),
            status="pending",
            real=self.is_real,
            session_id=safe(self.session_id),
            request_id=safe(request.metadata.request_id),
            reserved_tokens=request.token_reservation or 0,
        )
        if self.is_real:
            assert self.guard is not None and request.token_reservation is not None
            return self.guard.reserve(request.token_reservation, record=record)
        self.ledger.record_call(record)
        return record.id

    def _settle_attempt(
        self,
        call_id: str | None,
        usage: Usage,
        *,
        status: str,
        status_code: int | None,
        error_kind: str | None = None,
    ) -> GuardSettlement | None:
        if call_id is None:
            return None
        if self.is_real:
            assert self.guard is not None
            return self.guard.reconcile(
                call_id, usage=usage, status=status, status_code=status_code, error_kind=error_kind
            )
        assert self.ledger is not None
        self.ledger.finish_call(
            call_id,
            status=status,
            status_code=status_code,
            error_kind=error_kind,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            tokens_cached=usage.tokens_cached,
            total_tokens=usage.total_tokens,
        )
        return None

    @staticmethod
    def _failure_status(kind: ErrorKind) -> str:
        if kind.startswith("rate_limit_"):
            return "rate_limited"
        return "timeout" if kind in {"timeout", "cancelled"} else "error"

    async def _stream(
        self,
        response: httpx.Response,
        request: LLMRequest,
        snapshot: HeaderSnapshot,
        start: int,
        diagnostics: ProviderDiagnostics,
        state: ResponseState,
        context: EventContext,
        on_delta: DeltaCallback | None,
    ) -> LLMResult:
        seen: set[str] = set()

        async def deliver(delta: StreamDelta) -> None:
            if delta.what not in seen:
                seen.add(delta.what)
                self._emit(
                    context,
                    "first_delta",
                    call_id=request.metadata.call_id,
                    what=delta.what,
                    provider_elapsed_ms=delta.t_ms,
                )
            if on_delta is not None:
                try:
                    await on_delta(delta)
                except Exception:
                    raise DeltaCallbackError("Application delta callback failed") from None

        await consume_stream(
            response,
            request,
            state,
            diagnostics,
            clock=self.clock,
            start_ns=start,
            sanitizer=self.sanitizer,
            on_delta=deliver,
        )
        return LLMResult(
            text=state.text,
            model=request.model or self.model,
            response_model=state.response_model,
            completion_id=state.completion_id,
            provider_request_id=state.provider_request_id,
            status_code=response.status_code,
            usage=state.usage,
            usage_complete=state.usage.complete,
            header_snapshot=snapshot,
            provider_ttft_ms=state.first_content_ms,
            first_content_delta_ms=state.first_content_ms,
            first_reasoning_delta_ms=state.first_reasoning_ms,
            total_ms=(self.clock.monotonic_ns() - start) / 1_000_000,
            structured_output=validate_output(state.text, request),
            diagnostics=diagnostics,
        )

    def _nonstream(
        self,
        raw: bytes,
        request: LLMRequest,
        snapshot: HeaderSnapshot,
        start: int,
        diagnostics: ProviderDiagnostics,
        state: ResponseState,
    ) -> LLMResult:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Completion response must be an object")
        diagnostics.response_fields = [self.sanitizer.text(key) for key in body]
        usage = parse_usage(body, diagnostics)
        state.usage = usage
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Missing completion choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Missing message content")
        text = message["content"]
        if not text:
            raise ValueError("Empty completion response")
        metadata = body.get("x_groq")

        def identifier(value: object) -> str | None:
            return self.sanitizer.text(value) if isinstance(value, str) else None

        return LLMResult(
            text=text,
            model=request.model or self.model,
            response_model=identifier(body.get("model")),
            completion_id=identifier(body.get("id")),
            status_code=200,
            provider_request_id=identifier(metadata.get("id"))
            if isinstance(metadata, dict)
            else None,
            usage=usage,
            usage_complete=usage.complete,
            header_snapshot=snapshot,
            total_ms=(self.clock.monotonic_ns() - start) / 1_000_000,
            structured_output=validate_output(text, request),
            diagnostics=diagnostics,
        )

    async def complete(
        self, request: LLMRequest, *, on_delta: DeltaCallback | None = None
    ) -> LLMResult:
        start = self.clock.monotonic_ns()
        context = EventContext(
            rid=self.sanitizer.text(request.metadata.request_id),
            run=self.sanitizer.text(request.metadata.run_id),
            arrival_ns=request.metadata.arrival_ns
            if request.metadata.arrival_ns is not None
            else start,
        )
        snapshot = HeaderSnapshot(observed_ns=start)
        diagnostics = ProviderDiagnostics()
        state = ResponseState()
        raw_body = ""
        status: int | None = None
        call_id: str | None = None
        settled = False
        dispatched = False
        try:
            request = self._prepare(request, context)
            call_id = self._begin_attempt(request)
            self._emit(
                context,
                "call_start",
                call_id=request.metadata.call_id,
                kind=request.metadata.kind,
                attempt=request.metadata.attempt,
                model=request.model or self.model,
                source="real" if self.is_real else "stand-in",
            )
            auth = {}
            if self._api_key.get_secret_value():
                auth["Authorization"] = "Bearer " + self._api_key.get_secret_value()
            deadline = (
                asyncio.timeout(request.timeout_ms / 1000) if request.timeout_ms else nullcontext()
            )
            start = self.clock.monotonic_ns()  # Dispatch origin excludes guard/ledger setup.
            dispatched = True
            async with (
                deadline,
                self._client.stream(
                    "POST", "chat/completions", json=request.to_payload(self.model), headers=auth
                ) as response,
            ):
                status = response.status_code
                snapshot = header_snapshot(
                    response.headers, clock=self.clock, sanitizer=self.sanitizer
                )
                if not 200 <= status < 300:
                    raw = await response.aread()
                    try:
                        error_body = json.loads(raw)
                    except ValueError:
                        error_body = None
                    if isinstance(error_body, dict):
                        state.usage = parse_usage(error_body, diagnostics)
                    raise ProviderError(
                        classify_error(
                            status,
                            raw.decode("utf-8", "replace"),
                            snapshot,
                            rules=self.settings.error_rules,
                            sanitizer=self.sanitizer,
                        )
                    )
                if request.stream:
                    result = await self._stream(
                        response, request, snapshot, start, diagnostics, state, context, on_delta
                    )
                else:
                    raw = await response.aread()
                    raw_body = self.sanitizer.text(raw.decode("utf-8", "replace"), body=True)
                    if request.capture_response:
                        diagnostics.body_trunc = raw_body
                    result = self._nonstream(raw, request, snapshot, start, diagnostics, state)
                result.status_code = status
            result.total_ms = (self.clock.monotonic_ns() - start) / 1_000_000
            settled = True
            settlement = self._settle_attempt(
                call_id, result.usage, status="ok", status_code=status
            )
            if settlement is not None and settlement.cap_exceeded:
                raise self._error(
                    "budget_exceeded",
                    start,
                    snapshot,
                    usage=result.usage,
                    reason="Reported usage exceeds a cap or has inconsistent totals",
                )
            self._emit(
                context,
                "call_end",
                call_id=request.metadata.call_id,
                status=status,
                usage=result.usage.model_dump(),
                hdr=snapshot.headers,
                total_ms=result.total_ms,
                source="real" if self.is_real else "stand-in",
                model=result.model,
            )
            return result
        except asyncio.CancelledError:
            if not settled:
                self._settle_attempt(
                    call_id,
                    state.usage,
                    status="timeout",
                    status_code=status,
                    error_kind="cancelled",
                )
            self._emit(
                context,
                "call_error",
                call_id=request.metadata.call_id,
                reason="cancelled",
                status=status,
                hdr=snapshot.headers,
                usage=state.usage.model_dump(),
                total_ms=(self.clock.monotonic_ns() - start) / 1_000_000,
                source="real" if self.is_real else "stand-in",
            )
            cancelled = self._error("cancelled", start, snapshot, usage=state.usage)
            cancelled.info.status_code = status
            cancelled.info.safe_context.update(
                {
                    "dispatched": dispatched,
                    "headers_observed_ns": snapshot.observed_ns,
                    "first_reasoning_delta_ms": state.first_reasoning_ms,
                    "first_content_delta_ms": state.first_content_ms,
                }
            )
            raise ProviderCancelledError(cancelled.info) from None
        except Exception as exc:
            if isinstance(exc, ProviderError):
                error = exc
            elif isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                error = self._error("timeout", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, (httpx.TransportError, StreamDisconnectedError)):
                error = self._error("disconnect", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, DeltaCallbackError):
                error = self._error("callback", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, EventWriteError):
                error = self._error("persistence", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, LedgerError):
                error = self._error("persistence", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, RealCallDenied):
                error = self._error("budget_exceeded", start, snapshot, reason=str(exc))
            else:
                error = self._error("invalid_response", start, snapshot, reason=type(exc).__name__)
            if error.info.status_code is None:
                error.info.status_code = status
            error.info.usage = state.usage
            if not error.info.body_trunc:
                error.info.body_trunc = diagnostics.body_trunc or raw_body
            error.info.total_ms = (self.clock.monotonic_ns() - start) / 1_000_000
            error.info.safe_context.update(
                {
                    "dispatched": dispatched,
                    "headers_observed_ns": snapshot.observed_ns,
                    "first_reasoning_delta_ms": state.first_reasoning_ms,
                    "first_content_delta_ms": state.first_content_ms,
                }
            )
            if not settled:
                try:
                    self._settle_attempt(
                        call_id,
                        state.usage,
                        status=self._failure_status(error.kind),
                        status_code=error.status,
                        error_kind=error.kind,
                    )
                except LedgerError:
                    raise self._error(
                        "persistence",
                        start,
                        snapshot,
                        reason="Cannot finalise ledger; pending reservation retained",
                    ) from None
            if error.kind != "persistence":
                self._emit(
                    context,
                    "call_error",
                    call_id=request.metadata.call_id,
                    status=error.status,
                    reason=error.kind,
                    body_trunc=error.body_trunc,
                    hdr=error.headers,
                    total_ms=error.info.total_ms,
                    usage=state.usage.model_dump(),
                    source="real" if self.is_real else "stand-in",
                    safe_context=error.info.safe_context,
                )
            raise error from None
