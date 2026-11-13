"""P4 research-QA state machine: retrieve, draft, validate, one repair, release."""

import asyncio
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from relay.app.calls import CallRunner, Invocation, RequestCalls
from relay.app.delivery import _authorize
from relay.app.gate import EvidenceConflict, broadening_plan, evaluate, merge_results
from relay.app.interpret import interpret
from relay.app.models import Draft, Interpretation, Outcome, Question, State, ValidationResult
from relay.app.prompts import PromptError, Prompts
from relay.app.render import RenderedAnswer, RenderError, refusal_text, render_answer
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.app.state import StateMachine
from relay.app.validation import validate_evidence
from relay.app.verify import VerificationCallError, verify
from relay.control.ledger import Ledger
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriteError, EventWriter
from relay.providers.base import Provider
from relay.retrieval.models import RetrievalResult
from relay.settings import Setting
from relay.tools.executor import ToolExecutor


class PipelineDeadline(TimeoutError):
    """Logical monotonic deadline expired."""


class ToolStageError(RuntimeError):
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        super().__init__("Retrieval stage did not complete")


class ProviderStageError(RuntimeError):
    def __init__(self, invocation: Invocation) -> None:
        self.invocation = invocation
        super().__init__("Provider stage did not complete")


class PipelineCancelledError(asyncio.CancelledError):
    def __init__(self, outcome: Outcome) -> None:
        self.outcome = outcome
        super().__init__("Pipeline cancelled; partial accounting preserved")


class Run(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    question: Question
    calls: RequestCalls
    context: EventContext
    machine: StateMachine = Field(default_factory=StateMachine)
    interpretation: Interpretation | None = None
    retrieval: RetrievalResult | None = None
    queries: list[str] = Field(default_factory=list)
    draft: Draft | None = None
    validation: ValidationResult | None = None
    stage_ms: dict[str, float] = Field(default_factory=dict)
    tool_retries: int = 0


class Orchestrator:
    def __init__(
        self,
        provider: Provider,
        executor: ToolExecutor,
        *,
        model: str,
        clock: Clock | None = None,
        events: EventWriter | None = None,
        ledger: Ledger | None = None,
        config: PipelineSettings | None = None,
        run_id: str = "p4-local",
    ) -> None:
        self.clock = clock or SystemClock()
        self.config = config or pipeline_settings()
        self.events, self.executor, self.run_id = events, executor, run_id
        self.prompts = Prompts(self.config)
        self.runner = CallRunner(
            provider, model, clock=self.clock, config=self.config, ledger=ledger
        )

    def _emit(self, run: Run, ev: str, **fields: object) -> None:
        if self.events is not None:
            self.events.write(run.context, ev, **fields)

    def _move(self, run: Run, target: State) -> None:
        run.machine.validate(target)
        self._emit(run, "state_transition", previous=run.machine.state.value, state=target.value)
        run.machine.move(target)

    def _check(self, run: Run) -> None:
        if self.clock.monotonic_ns() >= run.calls.deadline_ns:
            raise PipelineDeadline("Logical request deadline expired")

    def _timing(self, run: Run, stage: str, started: int) -> None:
        run.stage_ms[stage] = (
            run.stage_ms.get(stage, 0) + (self.clock.monotonic_ns() - started) / 1e6
        )

    def _snapshot(
        self,
        run: Run,
        state: State,
        status: str,
        *,
        reason: str | None = None,
        text: str = "",
        rendered: RenderedAnswer | None = None,
        refusal: bool = False,
        releasable: bool = False,
    ) -> Outcome:
        attempts = run.calls.attempts
        supplied = (
            run.retrieval.hits
            if run.retrieval
            and any(a.dispatched and a.kind in {"draft", "repair"} for a in attempts)
            else []
        )
        return Outcome(
            request_id=run.question.request_id,
            run_id=self.run_id,
            text=text,
            state=state,
            status=status,
            failure_reason=reason,
            validated_refusal=refusal,
            refusal_reason=reason if refusal else None,
            releasable=releasable,
            citations=rendered.citations if rendered else [],
            claim_citations=rendered.claim_citations if rendered else {},
            draft_calls=sum(a.kind == "draft" and a.dispatched for a in attempts),
            verifier_calls=sum(a.kind == "verify" and a.dispatched for a in attempts),
            repair_calls=sum(a.kind == "repair" and a.dispatched for a in attempts),
            usage=run.calls.totals(),
            tool_retries=run.tool_retries,
            calls=attempts,
            stage_ms=run.stage_ms,
            provider_ms=sum(a.total_ms for a in attempts),
            provider_ttft_ms=next(
                (a.provider_ttft_ms for a in attempts if a.provider_ttft_ms is not None), None
            ),
            total_ms=(self.clock.monotonic_ns() - run.calls.arrival_ns) / 1e6,
            release_ready_at_ns=self.clock.monotonic_ns() if releasable else None,
            arrival_ns=run.calls.arrival_ns,
            deadline_ns=run.calls.deadline_ns,
            setting=run.question.setting or self.config.default_setting,
            supplied_chunk_ids=[h.chunk_id for h in supplied],
            retrieval_diagnostics=run.retrieval.diagnostics if run.retrieval else [],
            validation=run.validation,
            states=run.machine.history
            if run.machine.state == state
            else run.machine.history + [state],
        )

    def _complete(self, run: Run, state: State, status: str, **fields: object) -> Outcome:
        if fields.get("releasable"):
            self._check(run)
        outcome = self._snapshot(run, state, status, **fields)
        try:
            self._emit(
                run,
                "release_ready" if outcome.releasable else "done",
                state=state.value,
                status=status,
                failure_reason=outcome.failure_reason,
                total_ms=outcome.total_ms,
                provider_calls=outcome.provider_calls,
                usage_complete=outcome.usage.complete,
            )
            self._move(run, state)
        except EventWriteError:
            # Persistence failure is explicit and never authorizes an answer.
            run.machine.move(State.FAILED)
            return self._snapshot(run, State.FAILED, "error", reason="event_persistence")
        if outcome.releasable:
            _authorize(outcome)
        return outcome

    async def _retrieve(
        self,
        run: Run,
        query: str,
        *,
        paper_filter: str | None,
        paper_hint: str | None,
        numeric: bool,
    ) -> RetrievalResult:
        self._check(run)
        started = self.clock.monotonic_ns()
        try:
            result = await self.executor.execute(
                query,
                setting=run.question.setting or self.config.default_setting,
                paper_filter=paper_filter,
                paper_hint=paper_hint,
                numeric=numeric,
                deadline_ns=run.calls.deadline_ns,
                context=run.context,
                allow_retry=run.tool_retries < self.config.tool_retry_max,
            )
        finally:
            self._timing(run, "retrieval", started)
        self._move(run, State.RETRIEVED)
        run.tool_retries += max(0, result.attempt - 1)
        run.queries.append(query)
        run.retrieval = (
            result
            if run.retrieval is None
            else merge_results(run.retrieval, result)
            if result.status in {"ok", "empty"}
            else result
        )
        self._check(run)
        if result.status in {"error", "timeout"}:
            raise ToolStageError(result)
        return run.retrieval

    def _gate(self, run: Run) -> str:
        assert run.retrieval is not None and run.interpretation is not None
        started = self.clock.monotonic_ns()
        decision = evaluate(run.retrieval, run.interpretation, self.config)
        self._timing(run, "gate", started)
        self._move(run, State.GATE)
        self._emit(
            run,
            "gate_decision",
            decision=decision.status,
            coverage=decision.keyword_coverage,
            hits=len(run.retrieval.hits),
            source=decision.source,
        )
        return decision.status

    async def _evidence(self, run: Run) -> bool:
        assert run.interpretation is not None
        i = run.interpretation
        await self._retrieve(
            run,
            run.question.text,
            paper_filter=i.explicit_paper,
            paper_hint=i.paper_hint,
            numeric=i.numeric,
        )
        decision = self._gate(run)
        for step in range(1, self.config.broaden_max + 1):
            if decision == "sufficient":
                break
            assert run.retrieval is not None
            plan = broadening_plan(
                run.question.text, i, run.retrieval, step, run.queries, self.config
            )
            if plan is None:
                self._emit(run, "broaden_skipped", step=step, reason="no_meaningful_new_query")
                continue
            self._check(run)
            self._move(run, State.BROADEN)
            await self._retrieve(
                run,
                plan.query,
                paper_filter=plan.paper_filter,
                paper_hint=plan.paper_hint,
                numeric=plan.numeric,
            )
            decision = self._gate(run)
        assert run.retrieval is not None
        return bool(run.retrieval.hits)

    async def _draft(self, run: Run, repair: bool) -> bool:
        assert run.retrieval is not None
        self._check(run)
        self._move(run, State.REPAIRING if repair else State.DRAFTING)
        kind = "repair" if repair else "draft"
        self._emit(run, kind + "_start", supplied_hits=len(run.retrieval.hits))
        feedback = (
            [run.validation.failure_reason or "validation_failed"]
            if repair and run.validation
            else None
        )
        messages = self.prompts.draft(
            run.question.text,
            run.retrieval.hits,
            feedback=feedback,
            rejected=run.draft if repair else None,
        )
        started = self.clock.monotonic_ns()
        try:
            invocation = await self.runner.invoke(run.calls, kind, messages, Draft)
        finally:
            self._timing(run, kind, started)
        self._emit(
            run, kind + "_done", status=invocation.failure.kind if invocation.failure else "ok"
        )
        self._check(run)
        if invocation.failure is not None:
            if (
                invocation.failure.kind != "invalid_response"
                or invocation.failure.safe_context.get("origin") == "provider_boundary"
            ):
                raise ProviderStageError(invocation)
            run.draft = None
        else:
            assert invocation.result is not None
            try:
                run.draft = Draft.model_validate_json(invocation.result.text)
            except ValueError:
                run.draft = None
        if run.draft is not None and run.draft.status == "insufficient_evidence":
            run.validation = None
            return False
        self._move(run, State.VALIDATING)
        run.validation = (
            validate_evidence(run.draft, run.retrieval.hits, self.config)
            if run.draft
            else ValidationResult(
                status="fail_repairable", repairable=True, failure_reason="draft_schema_invalid"
            )
        )
        if run.validation.deterministic_passed:
            self._check(run)
            assert run.draft is not None
            started = self.clock.monotonic_ns()
            try:
                run.validation = await verify(
                    run.draft, run.validation, self.runner, run.calls, self.prompts
                )
            finally:
                self._timing(run, "verification", started)
            self._check(run)
        self._emit(
            run,
            "validated",
            status=run.validation.status,
            det="pass" if run.validation.deterministic_passed else "fail",
            llm="pass"
            if run.validation.verifier_passed
            else "invalid"
            if run.validation.status == "verify_invalid"
            else "fail",
            reason=run.validation.failure_reason,
        )
        return True

    async def _workflow(self, run: Run) -> Outcome:
        started = self.clock.monotonic_ns()
        run.interpretation = interpret(
            run.question.text, paper_scope=run.question.paper_scope, config=self.config
        )
        self._timing(run, "interpretation", started)
        self._move(run, State.INTERPRETED)
        self._emit(
            run,
            "interpreted",
            qtype=run.interpretation.qtype,
            numeric=run.interpretation.numeric,
            paper_hint=run.interpretation.paper_hint,
            out_of_scope_advisory=run.interpretation.looks_out_of_scope,
        )
        if not await self._evidence(run):
            return self._complete(
                run,
                State.REFUSED_LOCAL,
                "ok",
                text=refusal_text("local"),
                reason="no_evidence",
                refusal=True,
                releasable=True,
            )
        for generation in range(self.config.repair_max + 1):
            if not await self._draft(run, generation > 0):
                return self._complete(
                    run,
                    State.REFUSED_MODEL,
                    "ok",
                    text=refusal_text("model"),
                    reason="insufficient_evidence",
                    refusal=True,
                    releasable=True,
                )
            assert run.validation is not None
            if run.validation.status == "pass" and run.validation.verifier_passed:
                assert run.draft is not None and run.retrieval is not None
                self._check(run)
                started = self.clock.monotonic_ns()
                rendered = render_answer(run.draft, run.retrieval.hits, run.validation)
                self._timing(run, "rendering", started)
                return self._complete(
                    run,
                    State.RELEASED,
                    "ok",
                    text=rendered.text,
                    rendered=rendered,
                    releasable=True,
                )
            self._check(run)
            remaining = (run.calls.deadline_ns - self.clock.monotonic_ns()) / 1e6
            if (
                not run.validation.repairable
                or generation >= self.config.repair_max
                or remaining < self.config.repair_min_remaining_ms
            ):
                reason = (
                    "validation_failed_after_repair"
                    if generation
                    else "repair_deadline_insufficient"
                    if run.validation.repairable
                    and remaining < self.config.repair_min_remaining_ms
                    and self.config.repair_max
                    else "validation_failed"
                )
                return self._complete(
                    run,
                    State.ABSTAINED,
                    "error",
                    text=refusal_text("validation"),
                    reason=reason,
                    releasable=True,
                )
        raise RuntimeError("Unreachable bounded repair loop")

    async def handle(
        self,
        question: str | Question,
        setting: Setting | None = None,
        request_id: str | None = None,
        deadline_ms: int | None = None,
    ) -> Outcome:
        arrival = self.clock.monotonic_ns()
        data = question.model_dump() if isinstance(question, Question) else {"text": question}
        for field, value in (
            ("setting", setting),
            ("request_id", request_id),
            ("deadline_ms", deadline_ms),
        ):
            if value is not None:
                data[field] = value
        invalid = False
        try:
            parsed = Question.model_validate(data)
            if len(parsed.text) > self.config.max_question_chars:
                raise ValueError("Question exceeds configured application maximum")
        except ValueError:
            invalid = True
            fallback_id = (
                request_id if isinstance(request_id, str) and request_id.strip() else uuid4().hex
            )
            parsed = Question(
                text="?", request_id=fallback_id
            )  # Never retrieved for invalid input.
        budget = parsed.deadline_ms or self.config.deadline_ms
        book = RequestCalls(
            request_id=parsed.request_id,
            run_id=self.run_id,
            arrival_ns=arrival,
            deadline_ns=arrival + budget * 1_000_000,
        )
        run = Run(
            question=parsed,
            calls=book,
            context=EventContext(rid=parsed.request_id, run=self.run_id, arrival_ns=arrival),
        )
        try:
            self._emit(run, "arrived", setting=parsed.setting or self.config.default_setting)
            if invalid:
                return self._complete(run, State.FAILED, "error", reason="invalid_request")
            return await self._workflow(run)
        except (PipelineDeadline, ToolStageError) as exc:
            timeout = isinstance(exc, PipelineDeadline) or exc.result.status == "timeout"
            return self._complete(
                run,
                State.TIMED_OUT if timeout else State.FAILED,
                "timeout" if timeout else "error",
                reason="deadline"
                if isinstance(exc, PipelineDeadline)
                else "tool_" + exc.result.status,
            )
        except (ProviderStageError, VerificationCallError) as exc:
            failure = exc.invocation.failure
            assert failure is not None
            timeout = failure.kind == "timeout"
            status = (
                "timeout"
                if timeout
                else "rate_limited"
                if failure.kind.startswith("rate_limit_")
                else "error"
            )
            return self._complete(
                run,
                State.TIMED_OUT if timeout else State.FAILED,
                status,
                reason="provider_" + failure.kind,
            )
        except asyncio.CancelledError:
            outcome = self._complete(run, State.TIMED_OUT, "timeout", reason="cancelled")
            raise PipelineCancelledError(outcome) from None
        except EventWriteError:
            run.machine.move(State.FAILED)
            return self._snapshot(run, State.FAILED, "error", reason="event_persistence")
        except (PromptError, EvidenceConflict, RenderError) as exc:
            return self._complete(run, State.FAILED, "error", reason=type(exc).__name__)
        except Exception as exc:
            return self._complete(
                run, State.FAILED, "error", reason="pipeline_" + type(exc).__name__
            )
