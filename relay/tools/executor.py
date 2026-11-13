"""Narrow retrieval boundary with one explicit local transient retry."""

import asyncio
import random
import threading
from typing import Protocol

from pydantic import BaseModel

from relay.app.settings import PipelineSettings, ToolEnvironment, pipeline_settings
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriter
from relay.retrieval.models import RetrievalResult
from relay.retrieval.search import search_papers
from relay.settings import PaperId, Setting, retrieval_settings


class Search(Protocol):
    async def __call__(
        self,
        query: str,
        paper_filter: PaperId | None = None,
        top_k: int | None = None,
        setting: Setting = "C",
        *,
        paper_hint: PaperId | None = None,
        numeric: bool | None = None,
    ) -> RetrievalResult: ...


class ToolExecutor:
    def __init__(
        self,
        search: Search = search_papers,
        *,
        clock: Clock | None = None,
        rng: random.Random | None = None,
        events: EventWriter | None = None,
        config: PipelineSettings | None = None,
    ) -> None:
        self.search = search
        self.clock = clock or SystemClock()
        self.config = config or pipeline_settings()
        override = ToolEnvironment().tool_fault_prob
        self.fault_probability = self.config.tool_fault_prob if override is None else override
        self.rng = rng or random.Random(self.config.seed)
        self._lock = threading.Lock()
        self.events = events

    def _emit(self, context: EventContext, ev: str, **fields: object) -> None:
        if self.events is not None:
            self.events.write(context, ev, **fields)

    @staticmethod
    def _failure(status: str, query: str, setting: Setting, reason: str) -> RetrievalResult:
        return RetrievalResult(
            status=status,
            setting=setting,
            query_used=query,
            method=retrieval_settings().settings[setting].method,
            diagnostics=[reason],
        )

    async def _attempt(
        self,
        query: str,
        setting: Setting,
        paper_filter: PaperId | None,
        top_k: int | None,
        hint: PaperId | None,
        numeric: bool,
        timeout_ms: float,
    ) -> tuple[RetrievalResult, bool]:
        with self._lock:
            injected = self.rng.random() < self.fault_probability
        if injected:
            return self._failure(
                "error", query, setting, "Injected transient retrieval failure"
            ), True
        try:
            value = await asyncio.wait_for(
                self.search(query, paper_filter, top_k, setting, paper_hint=hint, numeric=numeric),
                timeout_ms / 1000,
            )
        except TimeoutError:
            return self._failure(
                "timeout",
                query,
                setting,
                "Tool timeout; underlying P1 CPU worker may still be finishing",
            ), False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._failure(
                "error", query, setting, f"Retrieval boundary failure ({type(exc).__name__})"
            ), True
        try:
            result = RetrievalResult.model_validate(
                value.model_dump() if isinstance(value, BaseModel) else value
            )
            if result.setting != setting or result.query_used != query:
                raise ValueError("Retrieval provenance mismatch")
            if (result.status == "ok") != bool(result.hits):
                raise ValueError("Retrieval status/hits mismatch")
        except (ValueError, TypeError):
            return self._failure("error", query, setting, "Malformed retrieval result"), False
        return result, result.status == "error"

    async def execute(
        self,
        query: str,
        *,
        setting: Setting,
        deadline_ns: int,
        context: EventContext,
        paper_filter: PaperId | None = None,
        top_k: int | None = None,
        paper_hint: PaperId | None = None,
        numeric: bool = False,
    ) -> RetrievalResult:
        start = self.clock.monotonic_ns()
        for attempt in range(1, self.config.tool_retry_max + 2):
            remaining = (deadline_ns - self.clock.monotonic_ns()) / 1e6
            if remaining <= 0:
                return self._failure("timeout", query, setting, "Logical request deadline exceeded")
            self._emit(
                context,
                "retrieval_start",
                setting=setting,
                attempt=attempt,
                query_chars=len(query),
                numeric=numeric,
            )
            result, retryable = await self._attempt(
                query,
                setting,
                paper_filter,
                top_k,
                paper_hint,
                numeric,
                min(remaining, self.config.tool_timeout_ms),
            )
            result.attempt = attempt
            result.took_ms = (self.clock.monotonic_ns() - start) / 1e6
            if self.clock.monotonic_ns() >= deadline_ns:
                result = self._failure(
                    "timeout", query, setting, "Logical request deadline exceeded"
                )
                result.attempt, result.took_ms = attempt, (self.clock.monotonic_ns() - start) / 1e6
                retryable = False
            self._emit(
                context,
                "retrieval_done",
                setting=setting,
                attempt=attempt,
                status=result.status,
                hits=len(result.hits),
                took_ms=result.took_ms,
                rerank_method=result.rerank_method,
            )
            if not retryable or attempt > self.config.tool_retry_max:
                return result
            if (
                deadline_ns - self.clock.monotonic_ns()
            ) / 1e6 < self.config.tool_retry_min_remaining_ms:
                return result
            self._emit(
                context, "tool_retry", reason="transient_retrieval", next_attempt=attempt + 1
            )
        raise RuntimeError("Unreachable bounded tool loop")
