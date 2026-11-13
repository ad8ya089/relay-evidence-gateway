"""One owner for P4 provider attempts, deadlines and ledger-reconciled usage."""

import asyncio
import math
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from relay.app.models import CallAttempt, TokenTotals
from relay.app.prompts import response_format
from relay.app.settings import PipelineSettings
from relay.control.ledger import Ledger, LedgerError
from relay.obs.clock import Clock
from relay.providers.base import (
    CallMetadata,
    ErrorKind,
    LLMRequest,
    LLMResult,
    Message,
    Provider,
    ProviderCancelledError,
    ProviderError,
    ProviderFailure,
    Usage,
)


class Invocation(BaseModel):
    result: LLMResult | None = None
    failure: ProviderFailure | None = None


class MalformedProviderResult(ValueError):
    """The provider returned an invalid value rather than a typed completion."""


class RequestCalls(BaseModel):
    request_id: str
    run_id: str
    arrival_ns: int
    deadline_ns: int
    attempts: list[CallAttempt] = Field(default_factory=list)

    def totals(self) -> TokenTotals:
        calls = [c for c in self.attempts if c.dispatched]
        fields = ("tokens_in", "tokens_out", "tokens_cached", "counted_tokens")
        totals: dict[str, object] = {}
        for field in fields:
            values = [getattr(c.usage, field) for c in calls]
            known = sum(v for v in values if v is not None)
            totals[field] = known if all(v is not None for v in values) else None
            totals["known_" + field] = known
        totals["unknown_usage_calls"] = sum(not c.usage_complete for c in calls)
        totals["complete"] = all(c.usage_complete for c in calls)
        return TokenTotals.model_validate(totals)


class CallRunner:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        clock: Clock,
        config: PipelineSettings,
        ledger: Ledger | None = None,
    ) -> None:
        self.provider, self.model, self.clock, self.config, self.ledger = (
            provider,
            model,
            clock,
            config,
            ledger,
        )

    @staticmethod
    def _failure(
        kind: ErrorKind,
        usage: Usage,
        previous: ProviderFailure | None,
        result: LLMResult | None = None,
    ) -> ProviderFailure:
        if previous is not None:
            return previous.model_copy(update={"kind": kind, "usage": usage})
        return ProviderFailure(
            kind=kind,
            usage=usage,
            status_code=result.status_code if result else None,
            headers=result.headers if result else {},
        )

    @staticmethod
    def _timing(value: object) -> float | None:
        if (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
        return None

    def _account(
        self, request: LLMRequest, usage: Usage, dispatched: bool
    ) -> tuple[Usage, bool, list[str]]:
        if self.ledger is None:
            return usage, False, ["No ledger injected; provider-reported usage only"]
        try:
            row = self.ledger.get_call(request.metadata.call_id)
        except LedgerError:
            return usage, False, ["missing_ledger_attempt"] if dispatched else []
        if (row.request_id, row.run_id, row.kind, row.model) != (
            request.metadata.request_id,
            request.metadata.run_id,
            request.metadata.kind,
            self.model,
        ):
            return usage, False, ["ledger_identity_mismatch"]
        names = ("tokens_in", "tokens_out", "tokens_cached", "total_tokens")
        reported = {name: getattr(usage, name) for name in names}
        persisted = {name: getattr(row, name) for name in names}
        diagnostics = []
        if reported != persisted:
            diagnostics.append("ledger_usage_mismatch")
        if row.status == "pending":
            diagnostics.append("ledger_attempt_not_finalized")
        return usage.model_copy(update=persisted), not diagnostics, diagnostics

    async def invoke(
        self,
        book: RequestCalls,
        kind: Literal["draft", "verify", "repair"],
        messages: list[Message],
        schema: type[BaseModel],
    ) -> Invocation:
        remaining = (book.deadline_ns - self.clock.monotonic_ns()) / 1e6
        if remaining <= 0:
            return Invocation(failure=ProviderFailure(kind="timeout"))
        verify = kind == "verify"
        stream = self.config.verify_stream if verify else self.config.draft_stream
        budget = min(
            remaining, self.config.verify_timeout_ms if verify else self.config.draft_timeout_ms
        )
        request = LLMRequest(
            model=self.model,
            messages=messages,
            stream=stream,
            response_format=response_format(schema, stream),
            validation_model=schema,
            max_completion_tokens=self.config.verify_max_completion_tokens
            if verify
            else self.config.draft_max_completion_tokens,
            reasoning_effort=self.config.reasoning_effort,
            timeout_ms=max(1, int(budget)),
            metadata=CallMetadata(
                request_id=book.request_id,
                run_id=book.run_id,
                kind=kind,
                arrival_ns=book.arrival_ns,
            ),
        )
        started = self.clock.monotonic_ns()
        result, failure, cancelled = None, None, None
        timeout_scope = asyncio.timeout(budget / 1000)
        try:
            async with timeout_scope:
                try:
                    received = await self.provider.complete(request)
                    if not isinstance(received, LLMResult):
                        raise MalformedProviderResult("Provider returned a malformed result")
                    try:
                        result = LLMResult.model_validate(received.model_dump())
                    except ValidationError:
                        raise MalformedProviderResult("Provider result schema is invalid") from None
                except ProviderCancelledError as exc:
                    failure = exc.info
                    raise
        except TimeoutError:
            failure = self._failure("timeout", failure.usage if failure else Usage(), failure)
        except ProviderError as exc:
            failure = exc.info
        except asyncio.CancelledError as exc:
            if timeout_scope.expired():
                failure = self._failure("timeout", failure.usage if failure else Usage(), failure)
            else:
                cancelled = exc
                if failure is None:
                    failure = ProviderFailure(kind="cancelled")
        except MalformedProviderResult:
            failure = ProviderFailure(
                kind="invalid_response", safe_context={"origin": "result_schema"}
            )
        except Exception as exc:
            failure = ProviderFailure(
                kind="invalid_response",
                safe_context={"reason": type(exc).__name__, "origin": "provider_boundary"},
            )
        usage = result.usage if result is not None else failure.usage if failure else Usage()
        dispatched = bool(failure.safe_context.get("dispatched", True)) if failure else True
        usage, reconciled, diagnostics = self._account(request, usage, dispatched)
        if self.ledger is not None and diagnostics:
            failure = self._failure("persistence", usage, failure, result)
        if self.clock.monotonic_ns() >= book.deadline_ns:
            failure = self._failure("timeout", usage, failure, result)
        timing_context = failure.safe_context if failure else {}
        first_content = (
            self._timing(result.first_content_delta_ms)
            if result
            else self._timing(timing_context.get("first_content_delta_ms"))
        )
        book.attempts.append(
            CallAttempt(
                call_id=request.metadata.call_id,
                kind=kind,
                dispatched=dispatched,
                usage=usage,
                usage_complete=usage.complete,
                total_ms=(self.clock.monotonic_ns() - started) / 1e6,
                provider_ttft_ms=self._timing(result.provider_ttft_ms) if result else first_content,
                first_content_delta_ms=first_content,
                first_reasoning_delta_ms=self._timing(result.first_reasoning_delta_ms)
                if result
                else self._timing(timing_context.get("first_reasoning_delta_ms")),
                status_code=result.status_code
                if result
                else failure.status_code
                if failure
                else None,
                failure_kind=failure.kind if failure else None,
                ledger_reconciled=reconciled,
                diagnostics=diagnostics,
            )
        )
        if cancelled is not None:
            raise cancelled
        return Invocation(result=None if failure else result, failure=failure)
