"""The only provider retry owner: bounded, jittered and globally budgeted."""

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from decimal import Decimal
from typing import TypeVar

from pydantic import BaseModel

from relay.app.calls import CallRunner, Invocation, RequestCalls
from relay.control.breaker import CircuitBreaker
from relay.control.limiter import Limiter, Need, QuotaDenied
from relay.control.scheduler import Kind, ProviderScheduler, SchedulerRejected
from relay.control.settings import RetrySettings
from relay.obs.clock import Clock
from relay.obs.events import EventContext, EventWriter, Sanitizer
from relay.providers.base import LLMRequest, Message, ProviderError, ProviderFailure
from relay.providers.errors import header_snapshot
from relay.retrieval.tokenizer import get_encoding

Sleep = Callable[[float], Awaitable[None]]
T = TypeVar("T")


class RetryBudget:
    def __init__(self, config: RetrySettings) -> None:
        self.config = config
        self.lock = asyncio.Lock()
        self.credit = Decimal(0)
        self.originals = self.retries = 0

    async def original(self) -> None:
        async with self.lock:
            self.originals += 1
            self.credit = min(
                Decimal(self.config.budget_capacity),
                self.credit + Decimal(str(self.config.budget_fraction)),
            )

    async def take(self) -> bool:
        async with self.lock:
            if self.credit < 1:
                return False
            self.credit -= 1
            self.retries += 1
            return True

    async def snapshot(self) -> dict[str, int | float]:
        async with self.lock:
            return {
                "tokens": float(self.credit),
                "original_attempts": self.originals,
                "retries": self.retries,
                "fraction": self.config.budget_fraction,
            }


class RetryController:
    def __init__(
        self,
        config: RetrySettings,
        clock: Clock,
        budget: RetryBudget,
        *,
        rng: random.Random | None = None,
        sleep: Sleep = asyncio.sleep,
        events: EventWriter | None = None,
        limiter: Limiter | None = None,
        scheduler: ProviderScheduler | None = None,
        breaker: CircuitBreaker | None = None,
        reservation_id: str | None = None,
        repair_need: Need | None = None,
        safety_tokens: int = 0,
        allow_unsafe_naive: bool = False,
    ) -> None:
        if config.policy == "naive_s7a" and not allow_unsafe_naive:
            raise ValueError("Unsafe S7a retries require explicit enablement")
        self.config, self.clock, self.budget = config, clock, budget
        self.rng, self.sleep, self.events = rng or random.Random(config.seed), sleep, events
        self.limiter, self.scheduler, self.breaker = limiter, scheduler, breaker
        self.reservation_id, self.repair_need, self.safety_tokens = (
            reservation_id,
            repair_need,
            safety_tokens,
        )

    async def allow_repair(self, book: RequestCalls) -> bool:
        if self.limiter is None:
            return True
        assert self.reservation_id and self.repair_need
        if self.breaker and (await self.breaker.snapshot(self.repair_need.model)).state == "open":
            return False
        return await self.limiter.extend(self.reservation_id, self.repair_need)

    @staticmethod
    def denied(reason: str) -> Invocation:
        return Invocation(
            failure=ProviderFailure(
                kind="budget_exceeded", safe_context={"dispatched": False, "control_reason": reason}
            )
        )

    def tokens(self, messages: list[Message], cap: int) -> int:
        framed = json.dumps(
            [m.model_dump() for m in messages], ensure_ascii=False, separators=(",", ":")
        )
        return len(get_encoding().encode(framed)) + cap + self.safety_tokens

    async def _once(
        self,
        runner: CallRunner,
        book: RequestCalls,
        kind: Kind,
        messages: list[Message],
        schema: type[BaseModel],
        attempt: int,
    ) -> Invocation:
        cap = (
            runner.config.verify_max_completion_tokens
            if kind == "verify"
            else runner.config.draft_max_completion_tokens
        )
        need = Need(model=runner.model, tokens=self.tokens(messages, cap), requests=1)
        tracked: list[str] = []
        invocation: Invocation | None = None

        async def prepare(request: LLMRequest) -> None:
            if self.clock.monotonic_ns() >= book.deadline_ns:
                raise ProviderError(
                    ProviderFailure(kind="timeout", safe_context={"dispatched": False})
                )
            if self.limiter:
                assert self.reservation_id
                if not await self.limiter.ensure(self.reservation_id, need):
                    raise ProviderError(
                        self.denied(
                            "daily_budget_exhausted"
                            if (await self.limiter.remaining(need.model)).tpd < need.tokens
                            else "projected_wait"
                        ).failure
                    )
                try:
                    await self.limiter.start_call(
                        self.reservation_id, request.metadata.call_id, need
                    )
                except QuotaDenied as exc:
                    raise ProviderError(self.denied(exc.reason).failure) from None
            tracked.append(request.metadata.call_id)

        started = self.clock.monotonic_ns()
        slot = self.scheduler.slot(kind, book.deadline_ns) if self.scheduler else nullcontext()
        entered = False
        try:
            await slot.__aenter__()
            entered = True
            if self.breaker and not await self.breaker.allow(runner.model, book.request_id):
                return self.denied("breaker_open")
            invocation = await runner.invoke_once(
                book,
                kind,
                messages,
                schema,
                attempt=attempt,
                token_reservation=need.tokens,
                before_dispatch=prepare,
            )
            return invocation
        except SchedulerRejected:
            return self.denied("deadline_infeasible")
        finally:

            async def settle() -> None:
                attempt_record = next((a for a in book.attempts if a.call_id in tracked), None)
                if attempt_record and self.limiter:
                    await self.limiter.reconcile(attempt_record.call_id, attempt_record.usage)
                    if invocation and invocation.result:
                        snapshot = invocation.result.header_snapshot
                    elif invocation and invocation.failure:
                        failure = invocation.failure
                        snapshot = header_snapshot(
                            failure.headers, clock=self.clock, sanitizer=Sanitizer()
                        )
                        stamp = failure.safe_context.get("headers_observed_ns")
                        snapshot.observed_ns = stamp if isinstance(stamp, int) else started
                    else:
                        snapshot = None
                    if snapshot:
                        await self.limiter.update_from_headers(runner.model, snapshot)
                    if invocation and invocation.failure:
                        await self.limiter.on_quota_error(
                            runner.model, invocation.failure.kind, invocation.failure.retry_after_ms
                        )
                if self.breaker:
                    if (
                        attempt_record
                        and attempt_record.dispatched
                        and attempt_record.failure_kind != "cancelled"
                    ):
                        await self.breaker.record(
                            runner.model, book.request_id, attempt_record.failure_kind
                        )
                    await self.breaker.release_probe(runner.model, book.request_id)

            cleanup = asyncio.create_task(settle())
            try:
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
            finally:
                if entered:
                    await slot.__aexit__(None, None, None)

    async def execute(
        self,
        runner: CallRunner,
        book: RequestCalls,
        kind: Kind,
        messages: list[Message],
        schema: type[BaseModel],
    ) -> Invocation:
        if self.config.policy == "naive_s7a":

            async def operation(attempt: int) -> Invocation:
                if attempt > 1:
                    book.retries += 1
                return await self._once(runner, book, kind, messages, schema, attempt)

            return await NaiveRetryPolicy(self.config).run(operation)
        await self.budget.original()
        attempt = 1
        while True:
            invocation = await self._once(runner, book, kind, messages, schema, attempt)
            failure = invocation.failure
            if (
                failure is None
                or failure.kind not in {"server", "timeout", "disconnect", "rate_limit_minute"}
                or book.retries >= self.config.max_per_request
            ):
                return invocation
            delay = max(
                self.rng.uniform(
                    0,
                    min(
                        self.config.max_backoff_ms, self.config.base_backoff_ms * 2 ** (attempt - 1)
                    ),
                ),
                failure.retry_after_ms or 0,
            )
            remaining = (book.deadline_ns - self.clock.monotonic_ns()) / 1e6
            if delay > remaining - self.config.safety_margin_ms:
                return invocation
            if not await self.budget.take():
                return invocation
            if self.limiter:
                assert self.reservation_id
                cap = (
                    runner.config.verify_max_completion_tokens
                    if kind == "verify"
                    else runner.config.draft_max_completion_tokens
                )
                if not await self.limiter.extend(
                    self.reservation_id,
                    Need(model=runner.model, tokens=self.tokens(messages, cap), requests=1),
                ):
                    return invocation
            if self.events:
                self.events.write(
                    EventContext(rid=book.request_id, run=book.run_id, arrival_ns=book.arrival_ns),
                    "retry_scheduled",
                    reason=failure.kind,
                    attempt=attempt + 1,
                    backoff_ms=delay,
                    retry_after_ms=failure.retry_after_ms,
                    remaining_deadline_ms=remaining,
                    budget=await self.budget.snapshot(),
                )
            before = self.clock.monotonic_ns()
            await self.sleep(delay / 1000)
            book.retry_wait_ms += (self.clock.monotonic_ns() - before) / 1e6
            if self.clock.monotonic_ns() >= book.deadline_ns:
                return invocation
            book.retries += 1
            attempt += 1


class NaiveRetryPolicy:
    """Explicit unsafe S7a demonstration: shield slow work and launch duplicates."""

    def __init__(self, config: RetrySettings) -> None:
        self.config = config

    async def run(self, operation: Callable[[int], Awaitable[T]]) -> T:
        tasks: list[asyncio.Task[T]] = []
        try:
            for attempt in range(1, self.config.naive_retries + 2):
                task = asyncio.create_task(operation(attempt))
                tasks.append(task)
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task), self.config.naive_timeout_ms / 1000
                    )
                except TimeoutError:
                    continue
            return await tasks[-1]
        finally:
            # Even this unsafe comparison owns every duplicate's cleanup/accounting.
            if asyncio.current_task().cancelling():
                for task in tasks:
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
