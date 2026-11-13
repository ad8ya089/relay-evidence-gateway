"""Predictive logical admission and bounded waiting, separate from provider slots."""

import asyncio
from typing import Literal

from pydantic import BaseModel

from relay.control.breaker import CircuitBreaker
from relay.control.limiter import Limiter, Need
from relay.control.settings import Policy
from relay.obs.clock import Clock
from relay.settings import Setting

Mode = Literal["normal", "pressure", "exhausted"]


class Reject(BaseModel):
    reason: str
    status_code: int = 429
    retry_after_s: float | None = None


class Admit(BaseModel):
    request_id: str
    mode: Mode
    need: Need
    arrival_ns: int
    deadline_ns: int
    projected_wait_ms: float
    expected_service_ms: int


class AdmissionRejected(RuntimeError):
    def __init__(self, rejection: Reject) -> None:
        self.rejection = rejection
        super().__init__(rejection.reason)


class Admission:
    def __init__(
        self, policy: Policy, limiter: Limiter, breaker: CircuitBreaker, clock: Clock
    ) -> None:
        self.policy, self.limiter, self.breaker, self.clock = policy, limiter, breaker, clock
        self.condition = asyncio.Condition()
        self.leases: dict[str, Admit] = {}
        self.active: set[str] = set()
        self.queue: list[str] = []
        self.peak_queue = 0

    async def mode(self, model: str) -> Mode:
        view = await self.limiter.remaining(model)
        breaker = await self.breaker.snapshot(model)
        if (
            view.tpd <= 0
            or view.rpd <= 0
            or view.blocked in {"provider_429_day", "provider_429_unknown"}
            or breaker.state == "open"
        ):
            return "exhausted"
        util = await self.limiter.utilisation(model)
        return (
            "pressure"
            if util["minute"] >= self.policy.pressure_minute_util
            or util["day"] >= self.policy.pressure_day_util
            else "normal"
        )

    def need(self, setting: Setting, model: str, mode: Mode, *, repair: bool = False) -> Need:
        estimate = self.policy.estimates[setting]
        limits = self.policy.pressure if mode == "pressure" else self.policy.normal
        return Need(
            model=model,
            tokens=estimate.draft_input
            + limits.draft_max_completion_tokens
            + estimate.verify_input
            + limits.verify_max_completion_tokens
            + self.policy.safety_tokens,
            requests=2,
        )

    async def admit(
        self, request_id: str, setting: Setting, model: str, deadline_ns: int
    ) -> Admit | Reject:
        async with self.condition:
            mode = await self.mode(model)
            need = self.need(setting, model, mode)
            view = await self.limiter.remaining(model)
            if view.blocked:
                return Reject(reason=view.blocked, retry_after_s=view.retry_after_s)
            if view.tpd < need.tokens or view.rpd < need.requests:
                return Reject(
                    reason="daily_budget_exhausted",
                    retry_after_s=self.limiter.local_daily_reset_s(),
                )
            breaker = await self.breaker.snapshot(model)
            if breaker.state == "open" or breaker.probe_inflight:
                return Reject(
                    reason="breaker_open", status_code=503, retry_after_s=breaker.retry_after_s
                )
            available = len(self.active) < self.policy.max_active_requests and not self.queue
            if not available and len(self.queue) >= self.policy.queue_capacity:
                return Reject(reason="queue_full")
            wait = await self.limiter.projected_wait(need)
            expected = self.policy.estimates[setting].expected_service_ms
            slot_wait = (
                0
                if available
                else expected * ((len(self.queue) // self.policy.max_active_requests) + 1)
            )
            if wait is None:
                return Reject(reason="projected_wait")
            wait = max(wait, slot_wait)
            if wait > self.policy.max_queue_wait_ms:
                return Reject(reason="projected_wait", retry_after_s=wait / 1000)
            if wait + expected > (deadline_ns - self.clock.monotonic_ns()) / 1e6:
                return Reject(reason="deadline_infeasible")
            if request_id in self.leases:
                return Reject(reason="request_in_progress", status_code=409)
            if not await self.limiter.reserve(request_id, need, max_wait_ms=wait):
                return Reject(reason="projected_wait")
            admitted = Admit(
                request_id=request_id,
                mode=mode,
                need=need,
                arrival_ns=self.clock.monotonic_ns(),
                deadline_ns=deadline_ns,
                projected_wait_ms=wait,
                expected_service_ms=expected,
            )
            self.leases[request_id] = admitted
            if available and wait == 0:
                self.active.add(request_id)
            else:
                if len(self.queue) >= self.policy.queue_capacity:
                    self.leases.pop(request_id)
                    await self.limiter.release(request_id)
                    return Reject(reason="queue_full")
                self.queue.append(request_id)
                self.peak_queue = max(self.peak_queue, len(self.queue))
            return admitted

    async def enter(self, admitted: Admit) -> float:
        async with self.condition:
            while admitted.request_id not in self.active:
                now = self.clock.monotonic_ns()
                patience = min(
                    admitted.deadline_ns,
                    admitted.arrival_ns + self.policy.max_queue_wait_ms * 1_000_000,
                )
                remaining = (patience - now) / 1e6
                wait = await self.limiter.wait_reserved(admitted.request_id)
                if remaining <= 0 or wait is None or wait > remaining:
                    raise AdmissionRejected(Reject(reason="projected_wait"))
                breaker = await self.breaker.snapshot(admitted.need.model)
                if breaker.state == "open":
                    raise AdmissionRejected(
                        Reject(
                            reason="breaker_open",
                            status_code=503,
                            retry_after_s=breaker.retry_after_s,
                        )
                    )
                if (
                    wait == 0
                    and len(self.active) < self.policy.max_active_requests
                    and self.queue[0] == admitted.request_id
                ):
                    self.queue.pop(0)
                    self.active.add(admitted.request_id)
                    break
                try:
                    await asyncio.wait_for(
                        self.condition.wait(), min(remaining, wait if wait else remaining) / 1000
                    )
                except TimeoutError:
                    pass
            return (self.clock.monotonic_ns() - admitted.arrival_ns) / 1e6

    async def release(self, request_id: str) -> None:
        async with self.condition:
            self.leases.pop(request_id, None)
            self.active.discard(request_id)
            if request_id in self.queue:
                self.queue.remove(request_id)
            await self.limiter.release(request_id)
            self.condition.notify_all()

    async def snapshot(self) -> dict[str, int]:
        async with self.condition:
            return {
                "queue_length": len(self.queue),
                "peak_queue_length": self.peak_queue,
                "active_logical_requests": len(self.active),
            }
