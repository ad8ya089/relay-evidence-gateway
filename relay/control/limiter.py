"""Atomic local quota reservations. Daily actual usage remains owned by P2."""

import asyncio
from datetime import UTC, timedelta

from pydantic import BaseModel, Field

from relay.control.ledger import Ledger, LedgerError
from relay.control.settings import Limits, Policy
from relay.obs.clock import Clock
from relay.providers.base import Usage


class Need(BaseModel):
    model: str
    tokens: int = Field(ge=0)
    requests: int = Field(ge=0)


class Reservation(BaseModel):
    need: Need
    ready_ns: int


class Charge(BaseModel):
    request_id: str
    model: str
    started_ns: int
    settled_ns: int | None = None
    tokens: int
    uncertain: bool = False


class Capacity(BaseModel):
    model: str
    limits: Limits
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    uncertain: bool
    outstanding_tokens: int
    outstanding_requests: int
    daily_used_tokens: int
    daily_attempts: int
    retry_after_s: float | None = None
    blocked: str | None = None


class QuotaDenied(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class Limiter:
    def __init__(self, policy: Policy, ledger: Ledger, clock: Clock, *, real: bool = False) -> None:
        self.policy, self.ledger, self.clock, self.real = policy, ledger, clock, real
        self.lock = asyncio.Lock()
        self.reservations: dict[str, Reservation] = {}
        self.charges: dict[str, Charge] = {}
        self._limits = dict(policy.limits)

    def _view(self, model: str, now: int, exclude: str | None = None) -> Capacity:
        limits = self._limits[model]
        reservations = [
            r.need
            for rid, r in self.reservations.items()
            if rid != exclude and r.need.model == model
        ]
        pending = [
            (cid, c) for cid, c in self.charges.items() if c.model == model and c.settled_ns is None
        ]
        outstanding_tokens = sum(r.tokens for r in reservations) + sum(c.tokens for _, c in pending)
        outstanding_requests = sum(r.requests for r in reservations)
        window_ns = self.policy.window_s * 1_000_000_000
        actual = [
            c
            for c in self.charges.values()
            if c.model == model and c.settled_ns is not None and c.settled_ns + window_ns > now
        ]
        requests = sum(
            c.model == model and c.started_ns + window_ns > now for c in self.charges.values()
        )
        usage = self.ledger.daily_usage(model, real=self.real)
        missing_tokens, missing_requests = 0, 0
        for cid, charge in pending:
            try:
                self.ledger.get_call(cid)
            except LedgerError:
                missing_tokens += charge.tokens
                missing_requests += 1
        day_tokens = usage.guard_charged_tokens + missing_tokens
        day_requests = usage.attempts + missing_requests
        unstarted_tokens = sum(r.tokens for r in reservations)
        uncertain = bool(usage.unknown_usage_count or any(c.uncertain for c in actual))
        return Capacity(
            model=model,
            limits=limits,
            rpm=max(0, limits.rpm - requests - outstanding_requests),
            tpm=max(0, limits.tpm - sum(c.tokens for c in actual) - outstanding_tokens),
            rpd=max(0, limits.rpd - day_requests - outstanding_requests),
            tpd=0
            if usage.unreserved_unknown_count
            else max(0, limits.tpd - day_tokens - unstarted_tokens),
            uncertain=uncertain,
            outstanding_tokens=outstanding_tokens,
            outstanding_requests=outstanding_requests,
            daily_used_tokens=day_tokens,
            daily_attempts=day_requests,
        )

    def _wait(self, need: Need, exclude: str | None = None) -> float | None:
        now = self.clock.monotonic_ns()
        view = self._view(need.model, now, exclude)
        if need.tokens > view.tpd or need.requests > view.rpd:
            return None
        if need.tokens > view.limits.tpm or need.requests > view.limits.rpm:
            return None
        candidates = {now}
        window = self.policy.window_s * 1_000_000_000
        for charge in self.charges.values():
            if charge.model == need.model:
                candidates.add(charge.started_ns + window)
                if charge.settled_ns is not None:
                    candidates.add(charge.settled_ns + window)
        for candidate in sorted(t for t in candidates if t >= now):
            projected = self._view(need.model, candidate, exclude)
            if need.tokens <= projected.tpm and need.requests <= projected.rpm:
                return (candidate - now) / 1e6
        return None  # In-flight work has no invented completion/reset time.

    async def projected_wait(self, need: Need) -> float | None:
        async with self.lock:
            return self._wait(need)

    async def reserve(self, request_id: str, need: Need, *, max_wait_ms: float = 0) -> bool:
        async with self.lock:
            if request_id in self.reservations:
                raise ValueError("Reservation IDs must be unique")
            wait = self._wait(need)
            if wait is None or wait > max_wait_ms:
                return False
            self.reservations[request_id] = Reservation(
                need=need.model_copy(), ready_ns=self.clock.monotonic_ns() + int(wait * 1e6)
            )
            return True

    async def extend(self, request_id: str, need: Need) -> bool:
        async with self.lock:
            reservation = self.reservations[request_id]
            combined = Need(
                model=need.model,
                tokens=reservation.need.tokens + need.tokens,
                requests=reservation.need.requests + need.requests,
            )
            if self._wait(combined, request_id) != 0:
                return False
            reservation.need = combined
            return True

    async def start_call(self, request_id: str, call_id: str, need: Need) -> None:
        async with self.lock:
            reservation = self.reservations[request_id]
            if (
                need.model != reservation.need.model
                or need.tokens > reservation.need.tokens
                or need.requests > reservation.need.requests
            ):
                raise QuotaDenied("reservation_insufficient")
            if self._wait(need, request_id) != 0:
                raise QuotaDenied("projected_wait")
            if call_id in self.charges:
                raise ValueError("Call ID already charged")
            reservation.need.tokens -= need.tokens
            reservation.need.requests -= need.requests
            self.charges[call_id] = Charge(
                request_id=request_id,
                model=need.model,
                started_ns=self.clock.monotonic_ns(),
                tokens=need.tokens,
            )

    async def reconcile(self, call_id: str, actual: Usage) -> None:
        async with self.lock:
            charge = self.charges[call_id]
            if charge.settled_ns is not None:
                return
            charge.settled_ns = self.clock.monotonic_ns()
            if actual.counted_tokens is None:
                charge.uncertain = True
            else:
                charge.tokens = actual.counted_tokens  # Excess is never clipped.

    async def release(self, request_id: str) -> None:
        async with self.lock:
            self.reservations.pop(request_id, None)  # Pending/unknown charges survive.

    async def remaining(self, model: str) -> Capacity:
        async with self.lock:
            return self._view(model, self.clock.monotonic_ns())

    async def utilisation(self, model: str) -> dict[str, float]:
        view = await self.remaining(model)
        return {"minute": 1 - view.tpm / view.limits.tpm, "day": 1 - view.tpd / view.limits.tpd}

    def local_daily_reset_s(self) -> float | None:
        if not self.policy.daily_reset_known:
            return None
        now = self.clock.utc_now().astimezone(UTC)
        end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (end - now).total_seconds()
