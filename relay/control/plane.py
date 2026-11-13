"""P5 execution policy around the unchanged P4 evidence-validation state machine."""

import asyncio
import random
from collections.abc import Callable

from relay.app.cache import AnswerCache, cache_key, for_request
from relay.app.calls import CallExecution
from relay.app.delivery import _authorize, approved
from relay.app.models import Outcome, Question, TokenTotals
from relay.app.orchestrator import Orchestrator
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.app.singleflight import FlightUnavailable, SingleFlight
from relay.control.admission import Admission, AdmissionRejected, Mode, Reject
from relay.control.breaker import CircuitBreaker
from relay.control.ledger import Ledger
from relay.control.limiter import Limiter
from relay.control.retry import RetryBudget, RetryController, Sleep
from relay.control.scheduler import ProviderScheduler
from relay.control.settings import Policy
from relay.obs.clock import Clock
from relay.obs.events import EventContext, EventWriter
from relay.providers.base import Provider
from relay.tools.executor import ToolExecutor

Factory = Callable[[PipelineSettings, CallExecution], Orchestrator]


class ControlPlane:
    def __init__(
        self,
        *,
        policy: Policy,
        clock: Clock,
        provider: Provider,
        ledger: Ledger,
        events: EventWriter,
        executor: ToolExecutor,
        model: str,
        corpus_hash: str,
        cache: AnswerCache,
        singleflight: SingleFlight,
        limiter: Limiter,
        admission: Admission,
        scheduler: ProviderScheduler,
        retry_budget: RetryBudget,
        breaker: CircuitBreaker,
        orchestrator_factory: Factory | None = None,
        rng: random.Random | None = None,
        sleep: Sleep = asyncio.sleep,
        allow_unsafe_naive: bool = False,
        run_id: str = "p5-local",
    ) -> None:
        if not corpus_hash or model not in policy.limits:
            raise ValueError("Known corpus identity and model quota policy required")
        self.policy, self.clock, self.provider, self.ledger, self.events = (
            policy,
            clock,
            provider,
            ledger,
            events,
        )
        self.executor, self.model, self.corpus_hash = executor, model, corpus_hash
        self.cache, self.singleflight, self.limiter, self.admission = (
            cache,
            singleflight,
            limiter,
            admission,
        )
        self.scheduler, self.retry_budget, self.breaker = scheduler, retry_budget, breaker
        self.scheduler.maximum = min(
            scheduler.maximum,
            policy.max_inflight_real
            if getattr(provider, "is_real", False)
            else policy.max_inflight_provider,
        )
        self.rng, self.sleep = rng or random.Random(policy.retry.seed), sleep
        self.factory = orchestrator_factory or self._orchestrator
        self.allow_unsafe_naive = allow_unsafe_naive
        self.run_id = run_id
        if policy.retry.policy == "naive_s7a" and not allow_unsafe_naive:
            raise ValueError("Unsafe naive retry demonstration must be explicitly enabled")

    def _orchestrator(self, config: PipelineSettings, execution: CallExecution) -> Orchestrator:
        return Orchestrator(
            self.provider,
            self.executor,
            model=self.model,
            clock=self.clock,
            events=self.events,
            ledger=self.ledger,
            config=config,
            execution=execution,
            run_id=self.run_id,
        )

    def _emit(self, request_id: str, arrival: int, event: str, **fields: object) -> None:
        self.events.write(
            EventContext(rid=request_id, run=self.run_id, arrival_ns=arrival), event, **fields
        )

    def pipeline_config(self, mode: Mode) -> PipelineSettings:
        limits = self.policy.pressure if mode == "pressure" else self.policy.normal
        return PipelineSettings.model_validate(
            pipeline_settings().model_dump()
            | {
                "deadline_ms": self.policy.deadline_ms,
                "evidence_limit": limits.top_k,
                "draft_max_completion_tokens": limits.draft_max_completion_tokens,
                "verify_max_completion_tokens": limits.verify_max_completion_tokens,
                "repair_max": limits.repair_max,
            }
        )

    async def _compute(
        self, question: Question, arrival: int, budget: int, key: str, use_cache: bool
    ) -> Outcome:
        if use_cache:
            cached = await self.cache.get(key, count=False)
            if cached:
                return for_request(
                    cached,
                    question.request_id,
                    arrival,
                    arrival + budget * 1_000_000,
                    cached=True,
                    now_ns=self.clock.monotonic_ns(),
                    mode=await self.admission.mode(self.model),
                )
        setting = question.setting or pipeline_settings().default_setting
        admitted = await self.admission.admit(
            question.request_id, setting, self.model, arrival + budget * 1_000_000
        )
        if isinstance(admitted, Reject):
            raise AdmissionRejected(admitted)
        try:
            self._emit(
                question.request_id,
                arrival,
                "admitted",
                mode=admitted.mode,
                wait_est_ms=admitted.projected_wait_ms,
                **await self.admission.snapshot(),
            )
            wait = await self.admission.enter(admitted)
            self._emit(question.request_id, arrival, "execution_start", queue_wait_ms=wait)
            execution = RetryController(
                self.policy.retry,
                self.clock,
                self.retry_budget,
                rng=self.rng,
                sleep=self.sleep,
                events=self.events,
                limiter=self.limiter,
                scheduler=self.scheduler,
                breaker=self.breaker,
                reservation_id=question.request_id,
                repair_need=self.admission.need(setting, self.model, admitted.mode, repair=True),
                safety_tokens=self.policy.safety_tokens,
                allow_unsafe_naive=self.allow_unsafe_naive,
                on_capacity_change=self.admission.wake,
            )
            app = self.factory(self.pipeline_config(admitted.mode), execution)
            outcome = await app.handle(
                question,
                setting=setting,
                request_id=question.request_id,
                deadline_ms=budget,
                arrival_ns=arrival,
                emit_arrived=False,
            )
            was_approved = approved(outcome)
            outcome.mode, outcome.queue_wait_ms = admitted.mode, wait
            if was_approved:
                _authorize(outcome)
            if use_cache:
                await self.cache.put(key, outcome)
            return outcome
        finally:
            cleanup = asyncio.create_task(self.admission.release(question.request_id))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def ask(
        self,
        question: Question,
        *,
        cache_enabled: bool | None = None,
        arrival_ns: int | None = None,
    ) -> Outcome | Reject:
        arrival = self.clock.monotonic_ns() if arrival_ns is None else arrival_ns
        budget = min(question.deadline_ms or self.policy.deadline_ms, self.policy.deadline_ms)
        deadline = arrival + budget * 1_000_000
        setting = question.setting or pipeline_settings().default_setting
        if setting not in self.policy.estimates:
            return Reject(reason="invalid_setting", status_code=422)
        use_cache = self.policy.cache.enabled and cache_enabled is not False
        key = cache_key(
            question.text,
            setting,
            self.model,
            self.policy.cache.prompt_version,
            self.corpus_hash,
            paper_scope=question.paper_scope,
        )
        self._emit(
            question.request_id,
            arrival,
            "arrived",
            server_arrival_monotonic=arrival,
            inflight=(await self.scheduler.snapshot()).inflight,
            mode=await self.admission.mode(self.model),
            **await self.admission.snapshot(),
        )
        if use_cache:
            cached = await self.cache.get(key)
            if cached:
                self._emit(question.request_id, arrival, "cache_hit")
                return for_request(
                    cached,
                    question.request_id,
                    arrival,
                    deadline,
                    cached=True,
                    now_ns=self.clock.monotonic_ns(),
                    mode=await self.admission.mode(self.model),
                )

        async def compute() -> Outcome:
            return await self._compute(question, arrival, budget, key, use_cache)

        try:
            if use_cache and self.policy.cache.singleflight:
                source, joined = await self.singleflight.run(key, compute, deadline)
            else:
                source, joined = await compute(), False
            if joined:
                self._emit(
                    question.request_id,
                    arrival,
                    "singleflight_join",
                    computation_request_id=source.request_id,
                )
            if approved(source):
                return for_request(
                    source,
                    question.request_id,
                    arrival,
                    deadline,
                    cached=source.cache_hit,
                    coalesced=joined,
                    now_ns=self.clock.monotonic_ns(),
                )
            outcome = source.model_copy(deep=True)
            outcome.request_id = question.request_id
            outcome.arrival_ns, outcome.deadline_ns = arrival, deadline
            outcome.coalesced = joined
            if joined:
                outcome.calls = []
                outcome.usage = TokenTotals()
                outcome.draft_calls = outcome.verifier_calls = outcome.repair_calls = (
                    outcome.retries
                ) = 0
            return outcome
        except AdmissionRejected as exc:
            rejection = exc.rejection
        except FlightUnavailable:
            rejection = Reject(reason="queue_full")
        except TimeoutError:
            rejection = Reject(reason="deadline_infeasible")
        self._emit(
            question.request_id,
            arrival,
            "rejected",
            reason=rejection.reason,
            retry_after_ms=None
            if rejection.retry_after_s is None
            else rejection.retry_after_s * 1000,
        )
        return rejection

    async def state(self) -> dict[str, object]:
        models = {}
        for model in self.policy.limits:
            models[model] = {
                "limiter": (await self.limiter.remaining(model)).model_dump(),
                "breaker": (await self.breaker.snapshot(model)).model_dump(),
                "mode": await self.admission.mode(model),
            }
        from relay.app.text import digest
        from relay.providers.guard import RealCallGuard

        guard = getattr(self.provider, "guard", None)
        proof = {"enabled": False, "permitted": False, "durable": False}
        if isinstance(guard, RealCallGuard):
            proof = {
                "enabled": True,
                "permitted": guard.settings.allow_real == "1" and guard.real_permission,
                "durable": guard.ledger is self.ledger,
                "remaining_requests": max(
                    0, guard.settings.real_request_cap - guard.requests_reserved
                ),
                "remaining_tokens": max(0, guard.settings.real_token_cap - guard.tokens_reserved),
            }
        return {
            "model": self.model,
            "provider_source": "real" if getattr(self.provider, "is_real", False) else "stand-in",
            "real_guard": proof,
            "policy_fingerprint": digest(self.policy.model_dump()),
            "mode": await self.admission.mode(self.model),
            **await self.admission.snapshot(),
            "provider": (await self.scheduler.snapshot()).model_dump(),
            "models": models,
            "retry_budget": await self.retry_budget.snapshot(),
            "cache": await self.cache.snapshot(),
            "singleflight": await self.singleflight.snapshot(),
        }
