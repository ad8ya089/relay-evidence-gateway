"""Bounded waiter coalescing with independent caller cancellation/deadlines."""

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from relay.app.models import Outcome
from relay.control.settings import CacheSettings
from relay.obs.clock import Clock


class FlightUnavailable(RuntimeError):
    pass


class Flight(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task: asyncio.Task[Outcome]
    waiters: int = 0


class SingleFlight:
    def __init__(self, config: CacheSettings, clock: Clock) -> None:
        self.config, self.clock = config, clock
        self.lock = asyncio.Lock()
        self.flights: dict[str, Flight] = {}
        self.joins = self.peak_waiters = 0
        self.closed = False

    async def run(
        self, key: str, compute: Callable[[], Awaitable[Outcome]], deadline_ns: int
    ) -> tuple[Outcome, bool]:
        async with self.lock:
            if (
                self.closed
                or sum(f.waiters for f in self.flights.values()) >= self.config.max_waiters
            ):
                raise FlightUnavailable("singleflight_capacity")
            flight = self.flights.get(key)
            joined = flight is not None
            if flight is not None and flight.task.cancelling():
                raise FlightUnavailable("computation_cancelling")
            if flight is None:
                flight = Flight(task=asyncio.create_task(compute()))
                self.flights[key] = flight
            else:
                self.joins += 1
            flight.waiters += 1
            self.peak_waiters = max(
                self.peak_waiters, sum(f.waiters for f in self.flights.values())
            )
        try:
            remaining = max(0, (deadline_ns - self.clock.monotonic_ns()) / 1e9)
            outcome = await asyncio.wait_for(asyncio.shield(flight.task), remaining)
            return outcome.model_copy(deep=True), joined
        finally:
            cleanup = False
            async with self.lock:
                flight.waiters -= 1
                if flight.waiters == 0:
                    cleanup = True
                    if not flight.task.done():
                        flight.task.cancel()
            if cleanup:
                await asyncio.gather(flight.task, return_exceptions=True)
                async with self.lock:
                    if self.flights.get(key) is flight and flight.waiters == 0:
                        self.flights.pop(key, None)

    async def close(self) -> None:
        async with self.lock:
            self.closed = True
            tasks = [f.task for f in self.flights.values()]
            for task in tasks:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def snapshot(self) -> dict[str, int]:
        async with self.lock:
            return {
                "operations": len(self.flights),
                "waiters": sum(f.waiters for f in self.flights.values()),
                "joins": self.joins,
                "peak_waiters": self.peak_waiters,
            }
