"""Atomic per-model rolling/daily quotas including lifetime token reservations."""

import math
import threading
from datetime import timedelta

from pydantic import BaseModel, Field

from relay.obs.clock import Clock
from standin.profile import Limits


class Charge(BaseModel):
    id: int
    started_ns: int
    day: str
    tokens: int
    pending: bool = True


class QuotaView(BaseModel):
    model: str
    usage: dict[str, int]
    remaining: dict[str, int]
    limits: dict[str, int]
    pending_tokens: int
    headers: dict[str, str]


class Admission(BaseModel):
    accepted: bool
    charge_id: int | None = None
    scopes: list[str] = Field(default_factory=list)
    retry_after_s: float | None = None
    view: QuotaView


class Quotas:
    def __init__(self, clock: Clock, window_s: int = 60) -> None:
        self.clock, self.window_ns = clock, window_s * 1_000_000_000
        self._lock = threading.RLock()
        self._models: dict[str, list[Charge]] = {}
        self._next_id = 0

    def reset(self) -> None:
        with self._lock:
            if any(c.pending for charges in self._models.values() for c in charges):
                raise ValueError("Cannot reset outstanding quota reservations")
            self._models.clear()
            # IDs remain unique so a delayed pre-reset finalizer cannot settle new work.

    def _active(self, model: str) -> list[Charge]:
        now, day = self.clock.monotonic_ns(), self.clock.utc_now().date().isoformat()
        charges = self._models.setdefault(model, [])
        charges[:] = [
            c for c in charges if c.pending or c.day == day or now - c.started_ns < self.window_ns
        ]
        return charges

    def _view(self, model: str, limits: Limits) -> QuotaView:
        charges, now = self._active(model), self.clock.monotonic_ns()
        utc = self.clock.utc_now()
        day = utc.date().isoformat()
        minute = [c for c in charges if now - c.started_ns < self.window_ns]
        daily = [c for c in charges if c.day == day]
        # A long-running reservation remains charged even after its start window/day expires.
        tokens_min = [c for c in charges if c.pending or now - c.started_ns < self.window_ns]
        tokens_day = [c for c in charges if c.pending or c.day == day]
        usage = {
            "rpm": len(minute),
            "tpm": sum(c.tokens for c in tokens_min),
            "rpd": len(daily),
            "tpd": sum(c.tokens for c in tokens_day),
        }
        numbers = limits.numbers()
        remaining = {name: max(0, cap - usage[name]) for name, cap in numbers.items()}
        reset_day = (
            (utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) - utc
        ).total_seconds()
        reset_token = max(
            (max(0, (c.started_ns + self.window_ns - now) / 1e9) for c in tokens_min), default=0
        )
        return QuotaView(
            model=model,
            usage=usage,
            limits=numbers,
            remaining=remaining,
            pending_tokens=sum(c.tokens for c in charges if c.pending),
            headers={
                "x-ratelimit-limit-requests": str(numbers["rpd"]),
                "x-ratelimit-remaining-requests": str(remaining["rpd"]),
                "x-ratelimit-reset-requests": f"{reset_day:.3f}s",
                "x-ratelimit-limit-tokens": str(numbers["tpm"]),
                "x-ratelimit-remaining-tokens": str(remaining["tpm"]),
                "x-ratelimit-reset-tokens": f"{reset_token:.3f}s",
            },
        )

    def view(self, model: str, limits: Limits) -> QuotaView:
        with self._lock:
            return self._view(model, limits)

    def _retry(self, model: str, scopes: list[str], need: int, view: QuotaView) -> float | None:
        charges = self._active(model)
        now, utc = self.clock.monotonic_ns(), self.clock.utc_now()
        waits = []
        for scope in scopes:
            required = need if scope in {"tpm", "tpd"} else 1
            if required > view.limits[scope]:
                return None  # This request cannot fit after any ordinary reset.
            if scope in {"rpd", "tpd"}:
                if scope == "tpd" and any(c.pending for c in charges):
                    return None
                waits.append(
                    (
                        (utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                        - utc
                    ).total_seconds()
                )
                continue
            deficit = view.usage[scope] + required - view.limits[scope]
            releases = sorted(
                (c.started_ns + self.window_ns, c.tokens if scope == "tpm" else 1)
                for c in charges
                if now - c.started_ns < self.window_ns and (scope == "rpm" or not c.pending)
            )
            recovered = 0
            for at, amount in releases:
                recovered += amount
                if recovered >= deficit:
                    waits.append(max(0, (at - now) / 1e9))
                    break
            else:
                return None  # Pending completion has no known recovery time.
        return max(waits) if waits else None

    def reserve(self, model: str, limits: Limits, tokens: int) -> Admission:
        if tokens <= 0:
            raise ValueError("A positive token reservation is required")
        with self._lock:
            view = self._view(model, limits)
            scopes = [
                name
                for name in view.limits
                if view.usage[name] + (tokens if name in {"tpm", "tpd"} else 1) > view.limits[name]
            ]
            if scopes:
                return Admission(
                    accepted=False,
                    scopes=scopes,
                    view=view,
                    retry_after_s=self._retry(model, scopes, tokens, view),
                )
            self._next_id += 1
            self._models[model].append(
                Charge(
                    id=self._next_id,
                    started_ns=self.clock.monotonic_ns(),
                    day=self.clock.utc_now().date().isoformat(),
                    tokens=tokens,
                )
            )
            return Admission(accepted=True, charge_id=self._next_id, view=self._view(model, limits))

    def settle(self, model: str, charge_id: int, actual_tokens: int | None) -> None:
        with self._lock:
            charge = next((c for c in self._models.get(model, []) if c.id == charge_id), None)
            if charge is None or not charge.pending:
                raise ValueError("Unknown or already settled quota reservation")
            if actual_tokens is not None:
                if actual_tokens < 0 or actual_tokens > charge.tokens:
                    raise ValueError("Actual simulated tokens must fit the reservation")
                charge.tokens = actual_tokens
            # Failed/aborted calls retain their conservative bound; no made-up zero usage.
            charge.pending = False


def retry_header(seconds: float | None) -> dict[str, str]:
    return {"retry-after": str(math.ceil(seconds))} if seconds is not None else {}
