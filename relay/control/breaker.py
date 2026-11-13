"""Per-model circuit breaker; only provider outage failures count."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from relay.control.settings import BreakerSettings
from relay.obs.clock import Clock


class BreakerState(BaseModel):
    state: Literal["closed", "open", "half_open"] = "closed"
    failures: list[int] = Field(default_factory=list)
    until_ns: int = 0
    probe_owner: str | None = None


class BreakerView(BaseModel):
    state: str
    failures: int
    probe_inflight: bool
    retry_after_s: float | None


class CircuitBreaker:
    def __init__(self, config: BreakerSettings, clock: Clock) -> None:
        self.config, self.clock = config, clock
        self.lock = asyncio.Lock()
        self.models: dict[str, BreakerState] = {}

    def _state(self, model: str) -> BreakerState:
        state = self.models.setdefault(model, BreakerState())
        now = self.clock.monotonic_ns()
        state.failures = [
            t for t in state.failures if now - t <= self.config.failure_window_ms * 1_000_000
        ]
        if state.state == "open" and now >= state.until_ns:
            state.state, state.probe_owner = "half_open", None
        return state

    async def allow(self, model: str, request_id: str) -> bool:
        async with self.lock:
            state = self._state(model)
            if state.state == "open":
                return False
            if state.state == "half_open":
                if state.probe_owner not in {None, request_id}:
                    return False
                state.probe_owner = request_id
            return True

    async def record(self, model: str, request_id: str, failure: str | None) -> None:
        async with self.lock:
            state = self._state(model)
            outage = failure in {"server", "timeout"} or (
                self.config.count_minute_429 and failure == "rate_limit_minute"
            )
            probe = state.state == "half_open" and state.probe_owner == request_id
            if probe:
                if failure is None:
                    state.state, state.failures, state.probe_owner = "closed", [], None
                else:
                    state.state, state.probe_owner = "open", None
                    state.until_ns = self.clock.monotonic_ns() + self.config.open_ms * 1_000_000
            elif state.state == "closed":
                if failure is None:
                    state.failures.clear()
                elif outage:
                    state.failures.append(self.clock.monotonic_ns())
                    if len(state.failures) >= self.config.consecutive_failures:
                        state.state = "open"
                        state.until_ns = self.clock.monotonic_ns() + self.config.open_ms * 1_000_000

    async def release_probe(self, model: str, request_id: str) -> None:
        async with self.lock:
            state = self._state(model)
            if state.probe_owner == request_id:
                state.probe_owner = None

    async def snapshot(self, model: str) -> BreakerView:
        async with self.lock:
            state = self._state(model)
            return BreakerView(
                state=state.state,
                failures=len(state.failures),
                probe_inflight=state.probe_owner is not None,
                retry_after_s=max(0, (state.until_ns - self.clock.monotonic_ns()) / 1e9)
                if state.state == "open"
                else None,
            )
