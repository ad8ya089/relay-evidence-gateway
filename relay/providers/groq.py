"""One raw-httpx adapter for loopback stand-ins and guarded external providers."""

import asyncio
import ipaddress
import json
from contextlib import nullcontext

import httpx
from pydantic import SecretStr

from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriteError, EventWriter, Sanitizer
from relay.providers.base import (
    DeltaCallback,
    ErrorKind,
    HeaderSnapshot,
    LLMRequest,
    LLMResult,
    ProviderDiagnostics,
    ProviderError,
    ProviderFailure,
    ResponseFormat,
    Usage,
)
from relay.providers.errors import classify_error, header_snapshot
from relay.providers.parsing import ResponseState, parse_usage, validate_output
from relay.providers.settings import ProviderSettings, provider_settings


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
    ) -> None:
        self.settings = settings if settings is not None else provider_settings()
        self.clock = clock if clock is not None else SystemClock()
        self.events = events
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
            raise ProviderError(
                ProviderFailure(
                    kind="auth",
                    safe_context={"reason": "External calls require guard/ledger integration"},
                )
            )
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
        if request.stream:
            raise ProviderError(ProviderFailure(kind="unsupported_combination"))
        return request

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
        try:
            request = self._prepare(request, context)
            self._emit(
                context,
                "call_start",
                call_id=request.metadata.call_id,
                kind=request.metadata.kind,
                attempt=request.metadata.attempt,
                model=request.model or self.model,
                source="stand-in",
            )
            auth = {}
            if self._api_key.get_secret_value():
                auth["Authorization"] = "Bearer " + self._api_key.get_secret_value()
            deadline = (
                asyncio.timeout(request.timeout_ms / 1000) if request.timeout_ms else nullcontext()
            )
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
                raw = await response.aread()
                raw_body = self.sanitizer.text(raw.decode("utf-8", "replace"), body=True)
                if request.capture_response or not 200 <= status < 300:
                    diagnostics.body_trunc = self.sanitizer.text(
                        raw.decode("utf-8", "replace"), body=True
                    )
                if not 200 <= status < 300:
                    raise ProviderError(
                        classify_error(
                            status,
                            raw.decode("utf-8", "replace"),
                            snapshot,
                            rules=self.settings.error_rules,
                            sanitizer=self.sanitizer,
                        )
                    )
                result = self._nonstream(raw, request, snapshot, start, diagnostics, state)
                result.status_code = status
            result.total_ms = (self.clock.monotonic_ns() - start) / 1_000_000
            self._emit(
                context,
                "call_end",
                call_id=request.metadata.call_id,
                status=status,
                usage=result.usage.model_dump(),
                hdr=snapshot.headers,
                total_ms=result.total_ms,
                source="stand-in",
                model=result.model,
            )
            return result
        except asyncio.CancelledError:
            self._emit(context, "call_error", call_id=request.metadata.call_id, reason="cancelled")
            raise
        except Exception as exc:
            if isinstance(exc, ProviderError):
                error = exc
            elif isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                error = self._error("timeout", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, httpx.TransportError):
                error = self._error("disconnect", start, snapshot, reason=type(exc).__name__)
            elif isinstance(exc, EventWriteError):
                error = self._error("persistence", start, snapshot, reason=type(exc).__name__)
            else:
                error = self._error("invalid_response", start, snapshot, reason=type(exc).__name__)
            if error.info.status_code is None:
                error.info.status_code = status
            error.info.usage = state.usage
            if not error.info.body_trunc:
                error.info.body_trunc = raw_body
            error.info.total_ms = (self.clock.monotonic_ns() - start) / 1_000_000
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
                )
            raise error from None
