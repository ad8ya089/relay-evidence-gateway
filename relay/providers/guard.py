"""Fail-closed P1 permission and session-budget foundation. No provider client."""

import sys
import threading
from collections.abc import Sequence
from uuid import uuid4

from pydantic import BaseModel

from relay.control.ledger import CallRecord, Ledger, SessionCapExceeded
from relay.providers.base import Usage
from relay.settings import SafetySettings

_PROCESS_SESSION_ID = uuid4().hex


class RealCallDenied(PermissionError):
    """Missing permission or exhausted session reservation budget."""


class GuardSettlement(BaseModel):
    charged_tokens: int
    requests: int
    usage_unknown: bool
    cap_exceeded: bool


class RealCallGuard:
    """Reserve worst-case tokens *before* a future adapter sends a request.

    With a ledger, pending reservations and observed usage are durable and shared
    by all guards using the same session. Unknown usage retains its reservation.
    The original standalone P1 mode remains conservative and is backward compatible.
    """

    def __init__(
        self,
        settings: SafetySettings | None = None,
        *,
        argv: Sequence[str] | None = None,
        ledger: Ledger | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings if settings is not None else SafetySettings()
        self.real_permission = "--real" in (sys.argv[1:] if argv is None else argv)
        self.ledger = ledger
        self.session_id = session_id or _PROCESS_SESSION_ID
        self._requests = 0
        self._tokens = 0
        self._pending: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def requests_reserved(self) -> int:
        return (
            self.ledger.session_usage(self.session_id).attempts if self.ledger else self._requests
        )

    @property
    def tokens_reserved(self) -> int:
        return (
            self.ledger.session_usage(self.session_id).guard_charged_tokens
            if self.ledger
            else self._tokens
        )

    def check_permissions(self) -> None:
        if self.settings.allow_real != "1" or not self.real_permission:
            raise RealCallDenied("Real calls require RELAY_ALLOW_REAL=1 AND process --real")

    def reserve(self, max_tokens: int, *, record: CallRecord | None = None) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must include a positive worst-case token reservation")
        with self._lock:
            self.check_permissions()
            if self.ledger is not None:
                if record is None or record.session_id != self.session_id:
                    raise RealCallDenied(
                        "Durable reservations require the guard's session metadata"
                    )
                try:
                    self.ledger.reserve_attempt(
                        record.model_copy(update={"reserved_tokens": max_tokens}),
                        token_cap=self.settings.real_token_cap,
                        request_cap=self.settings.real_request_cap,
                    )
                except SessionCapExceeded as exc:
                    raise RealCallDenied(str(exc)) from None
                return record.id
            if self.requests_reserved + 1 > self.settings.real_request_cap:
                raise RealCallDenied("RELAY_REAL_REQUEST_CAP exhausted")
            if self.tokens_reserved + max_tokens > self.settings.real_token_cap:
                raise RealCallDenied("RELAY_REAL_TOKEN_CAP would be exceeded")
            self._requests += 1
            self._tokens += max_tokens
            call_id = uuid4().hex
            self._pending[call_id] = max_tokens
            return call_id

    def reconcile(
        self,
        call_id: str,
        *,
        usage: Usage,
        status: str,
        status_code: int | None = None,
        error_kind: str | None = None,
    ) -> GuardSettlement:
        with self._lock:
            if self.ledger is not None:
                record = self.ledger.get_call(call_id)
                if record.session_id != self.session_id or not record.real:
                    raise RealCallDenied("Attempt does not belong to this real-call session")
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
                summary = self.ledger.session_usage(self.session_id)
                charged, requests = summary.guard_charged_tokens, summary.attempts
                inconsistent = bool(summary.inconsistent_usage_count)
            else:
                if call_id not in self._pending:
                    raise RealCallDenied("Unknown or already reconciled reservation")
                reserved = self._pending.pop(call_id)
                charge = usage.counted_tokens
                if charge is None:
                    charge = max(
                        reserved,
                        usage.total_tokens or 0,
                        (usage.tokens_in or 0) + (usage.tokens_out or 0),
                    )
                self._tokens += charge - reserved
                charged, requests = self._tokens, self._requests
                inconsistent = False
            return GuardSettlement(
                charged_tokens=charged,
                requests=requests,
                usage_unknown=usage.counted_tokens is None,
                cap_exceeded=inconsistent
                or charged > self.settings.real_token_cap
                or requests > self.settings.real_request_cap,
            )
