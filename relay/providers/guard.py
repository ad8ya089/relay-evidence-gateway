"""Fail-closed P1 permission and session-budget foundation. No provider client."""

import sys
import threading
from collections.abc import Sequence

from relay.settings import SafetySettings


class RealCallDenied(PermissionError):
    """Missing permission or exhausted session reservation budget."""


class RealCallGuard:
    """Reserve worst-case tokens *before* a future adapter sends a request.

    Reservations are deliberately conservative (never refunded). P2 must also record
    every real call in its durable SQLite ledger; this class never performs I/O.
    Share one instance per process/session; it is safe across threads.
    """

    def __init__(
        self, settings: SafetySettings | None = None, *, argv: Sequence[str] | None = None
    ) -> None:
        self.settings = settings if settings is not None else SafetySettings()
        self.real_permission = "--real" in (sys.argv[1:] if argv is None else argv)
        self.requests_reserved = 0
        self.tokens_reserved = 0
        self._lock = threading.Lock()

    def reserve(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must include a positive worst-case token reservation")
        with self._lock:
            if self.settings.allow_real != "1" or not self.real_permission:
                raise RealCallDenied("Real calls require RELAY_ALLOW_REAL=1 AND process --real")
            if self.requests_reserved + 1 > self.settings.real_request_cap:
                raise RealCallDenied("RELAY_REAL_REQUEST_CAP exhausted")
            if self.tokens_reserved + max_tokens > self.settings.real_token_cap:
                raise RealCallDenied("RELAY_REAL_TOKEN_CAP would be exceeded")
            self.requests_reserved += 1
            self.tokens_reserved += max_tokens
