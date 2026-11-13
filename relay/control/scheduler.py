"""Bounded provider concurrency with completion priority and draft aging."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict

from relay.control.settings import SchedulerSettings
from relay.obs.clock import Clock

Kind = Literal["draft", "verify", "repair"]


class SchedulerRejected(RuntimeError):
    pass


class Waiter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sequence: int
    kind: Kind
    arrived_ns: int
    deadline_ns: int
    future: asyncio.Future[None]


class SchedulerView(BaseModel):
    inflight: int
    waiters: int
    peak_waiters: int


class ProviderScheduler:
    def __init__(self, maximum: int, config: SchedulerSettings, clock: Clock) -> None:
        self.maximum, self.config, self.clock = maximum, config, clock
        self.lock = asyncio.Lock()
        self.inflight = self.peak_waiters = self.sequence = self.priority_burst = 0
        self.waiters: list[Waiter] = []

    def _drain(self) -> None:
        now = self.clock.monotonic_ns()
        live = []
        for waiter in self.waiters:
            if waiter.future.cancelled():
                continue
            if waiter.deadline_ns <= now:
                waiter.future.set_exception(SchedulerRejected("deadline_infeasible"))
            else:
                live.append(waiter)
        self.waiters = live
        while self.waiters and self.inflight < self.maximum:
            aged = [
                w for w in self.waiters if now - w.arrived_ns >= self.config.aging_ms * 1_000_000
            ]
            drafts = [w for w in self.waiters if w.kind == "draft"]
            if aged:
                selected = min(aged, key=lambda w: w.sequence)
            elif drafts and self.priority_burst >= self.config.max_priority_burst:
                selected = min(drafts, key=lambda w: w.sequence)
            else:
                selected = min(
                    self.waiters, key=lambda w: (self.config.priorities[w.kind], w.sequence)
                )
            self.waiters.remove(selected)
            self.priority_burst = 0 if selected.kind == "draft" else self.priority_burst + 1
            self.inflight += 1
            selected.future.set_result(None)

    @asynccontextmanager
    async def slot(self, kind: Kind, deadline_ns: int) -> AsyncIterator[None]:
        async with self.lock:
            if deadline_ns <= self.clock.monotonic_ns():
                raise SchedulerRejected("deadline_infeasible")
            if len(self.waiters) >= self.config.max_waiters:
                raise SchedulerRejected("provider_queue_full")
            waiter = Waiter(
                sequence=self.sequence,
                kind=kind,
                arrived_ns=self.clock.monotonic_ns(),
                deadline_ns=deadline_ns,
                future=asyncio.get_running_loop().create_future(),
            )
            self.sequence += 1
            self.waiters.append(waiter)
            self._drain()
            self.peak_waiters = max(self.peak_waiters, len(self.waiters))
        acquired = False
        try:
            remaining = (deadline_ns - self.clock.monotonic_ns()) / 1e9
            async with asyncio.timeout(max(0, remaining)):
                await waiter.future
            acquired = True
            yield
        except TimeoutError:
            raise SchedulerRejected("deadline_infeasible") from None
        finally:
            async with self.lock:
                if waiter in self.waiters:
                    self.waiters.remove(waiter)
                elif acquired or (
                    waiter.future.done()
                    and not waiter.future.cancelled()
                    and waiter.future.exception() is None
                ):
                    self.inflight -= 1
                self._drain()

    async def snapshot(self) -> SchedulerView:
        async with self.lock:
            return SchedulerView(
                inflight=self.inflight, waiters=len(self.waiters), peak_waiters=self.peak_waiters
            )
