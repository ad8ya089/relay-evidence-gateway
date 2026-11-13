"""Atomic local quota reservations. Daily actual usage remains owned by P2."""

import asyncio
import math
import re
from datetime import UTC, timedelta

from pydantic import BaseModel, Field

from relay.control.ledger import Ledger, LedgerError
from relay.control.settings import Limits, Policy
from relay.obs.clock import Clock
from relay.providers.base import HeaderSnapshot, Usage


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


class Observation(BaseModel):
    observed_ns: int
    remaining: int | None
    reset_ns: int | None
    day: str
    local_tokens: int
    local_requests: int


def reset_duration_s(value: str | None) -> float | None:
    """Parse a complete provider duration; malformed/unknown resets stay unknown."""
    if value is None:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value.strip())
    if not parts or "".join(n + unit for n, unit in parts) != value.strip():
        return None
    result = sum(float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit] for n, unit in parts)
    return result if math.isfinite(result) else None


class Limiter:
    def __init__(self, policy: Policy, ledger: Ledger, clock: Clock, *, real: bool = False) -> None:
        self.policy, self.ledger, self.clock, self.real = policy, ledger, clock, real
        self.lock = asyncio.Lock()
        self.reservations: dict[str, Reservation] = {}
        self.charges: dict[str, Charge] = {}
        self._limits = dict(policy.limits)
        self.observations: dict[str, dict[str, Observation]] = {}
        self.cooling: dict[str, int] = {}
        self.blocked: dict[str, tuple[str, int | None]] = {}
        self.total_tokens: dict[str, int] = {}
        self.total_requests: dict[str, int] = {}

    def _view(self, model: str, now: int, exclude: str | None = None) -> Capacity:
        cutoff = self.clock.monotonic_ns() - self.policy.window_s * 1_000_000_000
        self.charges = {
            cid: c
            for cid, c in self.charges.items()
            if c.settled_ns is None or c.settled_ns > cutoff or c.started_ns > cutoff
        }
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
        view = Capacity(
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
        for scope, observation in self.observations.get(model, {}).items():
            if observation.reset_ns is not None and now >= observation.reset_ns:
                continue
            if (
                scope == "rpd"
                and self.policy.daily_reset_known
                and observation.day != self.clock.utc_now().date().isoformat()
            ):
                continue
            if observation.remaining is None:
                continue
            spent = (
                self.total_tokens.get(model, 0) - observation.local_tokens
                if scope == "tpm"
                else self.total_requests.get(model, 0) - observation.local_requests
            )
            reserved = outstanding_tokens if scope == "tpm" else outstanding_requests + len(pending)
            setattr(
                view,
                scope,
                min(getattr(view, scope), max(0, observation.remaining - spent - reserved)),
            )
        until = self.cooling.get(model, 0)
        if until > now:
            view.blocked, view.retry_after_s = "provider_429_minute", (until - now) / 1e9
        if model in self.blocked:
            reason, expiry = self.blocked[model]
            if expiry is None or now < expiry:
                view.blocked = reason
                view.retry_after_s = None if expiry is None else (expiry - now) / 1e9
        return view

    def _wait(self, need: Need, exclude: str | None = None) -> float | None:
        now = self.clock.monotonic_ns()
        view = self._view(need.model, now, exclude)
        if view.blocked and view.retry_after_s is None:
            return None
        if need.tokens > view.tpd or need.requests > view.rpd:
            return None
        if need.tokens > view.limits.tpm or need.requests > view.limits.rpm:
            return None
        candidates = {now}
        candidates.add(self.cooling.get(need.model, now))
        for observation in self.observations.get(need.model, {}).values():
            if observation.reset_ns is not None:
                candidates.add(observation.reset_ns)
        if need.model in self.blocked and self.blocked[need.model][1] is not None:
            candidates.add(self.blocked[need.model][1])
        window = self.policy.window_s * 1_000_000_000
        for charge in self.charges.values():
            if charge.model == need.model:
                candidates.add(charge.started_ns + window)
                if charge.settled_ns is not None:
                    candidates.add(charge.settled_ns + window)
        for candidate in sorted(t for t in candidates if t >= now):
            projected = self._view(need.model, candidate, exclude)
            if (
                not projected.blocked
                and need.tokens <= projected.tpm
                and need.requests <= projected.rpm
            ):
                return (candidate - now) / 1e6
        return None  # In-flight work has no invented completion/reset time.

    async def projected_wait(self, need: Need) -> float | None:
        async with self.lock:
            return self._wait(need)

    async def wait_reserved(self, request_id: str) -> float | None:
        async with self.lock:
            return self._wait(self.reservations[request_id].need, request_id)

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

    async def ensure(self, request_id: str, need: Need) -> bool:
        async with self.lock:
            reservation = self.reservations[request_id]
            required = Need(
                model=need.model,
                tokens=max(need.tokens, reservation.need.tokens),
                requests=max(need.requests, reservation.need.requests),
            )
            if self._wait(required, request_id) != 0:
                return False
            reservation.need = required
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
            self.total_requests[need.model] = self.total_requests.get(need.model, 0) + need.requests

    async def reconcile(self, call_id: str, actual: Usage) -> None:
        async with self.lock:
            charge = self.charges.get(call_id)
            if charge is None:
                return  # A pruned, already-finalized attempt cannot be counted again.
            if charge.settled_ns is not None:
                return
            charge.settled_ns = self.clock.monotonic_ns()
            if actual.counted_tokens is None:
                charge.uncertain = True
            else:
                charge.tokens = actual.counted_tokens  # Excess is never clipped.
            self.total_tokens[charge.model] = self.total_tokens.get(charge.model, 0) + charge.tokens

    async def release(self, request_id: str) -> None:
        async with self.lock:
            self.reservations.pop(request_id, None)  # Pending/unknown charges survive.

    async def remaining(self, model: str) -> Capacity:
        async with self.lock:
            return self._view(model, self.clock.monotonic_ns())

    async def update_from_headers(self, model: str, snapshot: HeaderSnapshot) -> None:
        async with self.lock:
            if snapshot.observed_ns > self.clock.monotonic_ns():
                return
            observations = self.observations.setdefault(model, {})
            for scope, limit, remaining, suffix in (
                ("tpm", snapshot.limit_tokens_tpm, snapshot.remaining_tokens_tpm, "tokens"),
                ("rpd", snapshot.limit_requests_rpd, snapshot.remaining_requests_rpd, "requests"),
            ):
                if limit is None and remaining is None:
                    continue
                previous = observations.get(scope)
                if previous and snapshot.observed_ns < previous.observed_ns:
                    continue
                duration = reset_duration_s(snapshot.headers.get("x-ratelimit-reset-" + suffix))
                expiry = (
                    snapshot.observed_ns + int(duration * 1e9) if duration is not None else None
                )
                if scope == "tpm" and expiry is None:
                    expiry = snapshot.observed_ns + self.policy.window_s * 1_000_000_000
                if (
                    previous
                    and (previous.reset_ns is None or self.clock.monotonic_ns() < previous.reset_ns)
                    and previous.day == self.clock.utc_now().date().isoformat()
                ):
                    spent = (
                        self.total_tokens.get(model, 0) - previous.local_tokens
                        if scope == "tpm"
                        else self.total_requests.get(model, 0) - previous.local_requests
                    )
                    previous_remaining = (
                        max(0, previous.remaining - spent)
                        if previous.remaining is not None
                        else None
                    )
                    remaining = (
                        previous_remaining
                        if remaining is None
                        else min(remaining, previous_remaining)
                        if previous_remaining is not None
                        else remaining
                    )
                if limit is not None and limit > 0:
                    if previous and snapshot.observed_ns == previous.observed_ns:
                        limit = min(limit, getattr(self._limits[model], scope))
                    self._limits[model] = self._limits[model].model_copy(update={scope: limit})
                observations[scope] = Observation(
                    observed_ns=snapshot.observed_ns,
                    remaining=remaining,
                    reset_ns=expiry,
                    day=self.clock.utc_now().date().isoformat(),
                    local_tokens=self.total_tokens.get(model, 0),
                    local_requests=self.total_requests.get(model, 0),
                )

    async def on_quota_error(self, model: str, kind: str, retry_after_ms: float | None) -> None:
        async with self.lock:
            now = self.clock.monotonic_ns()
            if kind == "rate_limit_minute":
                self.cooling[model] = max(
                    self.cooling.get(model, 0),
                    now
                    + int(
                        (
                            retry_after_ms
                            if retry_after_ms is not None
                            else self.policy.window_s * 1000
                        )
                        * 1e6
                    ),
                )
            elif kind in {"rate_limit_day", "rate_limit_unknown"}:
                observation = self.observations.get(model, {}).get("rpd")
                expiry = observation.reset_ns if observation else None
                # Retry-After alone cannot establish daily/unknown quota recovery.
                self.blocked[model] = (
                    "provider_429_day" if kind == "rate_limit_day" else "provider_429_unknown",
                    expiry,
                )

    async def utilisation(self, model: str) -> dict[str, float]:
        view = await self.remaining(model)
        return {"minute": 1 - view.tpm / view.limits.tpm, "day": 1 - view.tpd / view.limits.tpd}

    def local_daily_reset_s(self) -> float | None:
        if not self.policy.daily_reset_known:
            return None
        now = self.clock.utc_now().astimezone(UTC)
        end = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (end - now).total_seconds()
