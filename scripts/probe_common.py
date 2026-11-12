"""Shared guarded probe planning, safe persistence and attempt recording."""

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field

from relay.control.ledger import Ledger
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriter, Sanitizer
from relay.providers.base import (
    CallMetadata,
    LLMRequest,
    Message,
    ProviderCancelledError,
    ProviderError,
    Usage,
)
from relay.providers.groq import GroqCompatibleProvider, is_loopback_url
from relay.providers.guard import RealCallGuard
from relay.retrieval.tokenizer import get_encoding
from relay.settings import SafetySettings

Sleep = Callable[[float], Awaitable[None]]


class ProbeError(RuntimeError):
    """Refused/aborted experiment; messages contain no credentials."""


class AccountLimits(BaseModel):
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)
    tpd: int | None = Field(default=None, gt=0)


class ProbeSettings(BaseModel):
    model: str
    real_base_url: str
    local_base_url: str
    reasoning_effort: str
    shapes_max_completion_tokens: int = Field(gt=0)
    limit_max_completion_tokens: int = Field(gt=0)
    reservation_overhead_tokens: int = Field(ge=0)
    rpm_requests: int = Field(gt=0, le=40)
    rpm_rates: list[float] = Field(min_length=1)
    tpm_requests: int = Field(ge=2, le=3)
    tpm_input_tokens: int = Field(gt=0)
    concurrency_requests: int = Field(gt=0, le=8)
    recovery_requests: int = Field(ge=0)
    recovery_interval_s: float = Field(ge=0, allow_inf_nan=False)
    max_recovery_wait_s: float = Field(ge=0, allow_inf_nan=False)
    daily_safety_fraction: float = Field(gt=0, lt=1)
    account_limits: AccountLimits
    remaining_day_requests: int | None = Field(default=None, gt=0)
    remaining_day_tokens: int | None = Field(default=None, gt=0)


@lru_cache(maxsize=1)
def probe_settings() -> ProbeSettings:
    return ProbeSettings.model_validate(yaml.safe_load(Path("config/probes.yaml").read_text()))


class PhasePlan(BaseModel):
    name: str
    max_requests: int
    estimated_tokens: int
    concurrency: int = 1
    offered_rates_per_minute: list[float] = Field(default_factory=list)


class ProbePlan(BaseModel):
    mode: Literal["dry-run", "local", "real"]
    model: str
    base_url: str
    phases: list[PhasePlan]
    max_requests: int
    estimated_tokens: int
    account_limits: AccountLimits
    remaining_day_requests: int | None = None
    remaining_day_tokens: int | None = None
    daily_safety_fraction: float | None = None
    account_limits_status: str = "unverified; fill from account evidence"
    session_request_cap: int
    session_token_cap: int
    token_estimate_status: str = "o200k_base estimate plus assumed overhead; not provider usage"


class ProbeRecord(BaseModel):
    phase: str
    request_id: str
    call_id: str
    source: Literal["real", "stand-in"]
    observed_against: Literal["Groq", "local service (not Groq)"]
    observed_at_utc: str
    status_code: int | None
    outcome: Literal["ok", "error"]
    error_kind: str | None = None
    dispatched: bool
    estimated_tokens: int | None
    usage: Usage
    headers: dict[str, str]
    retry_after_ms: float | None = None
    header_observed_ns: int | None = None
    total_ms: float
    provider_ttft_ms: float | None = None
    first_reasoning_delta_ms: float | None = None
    first_content_delta_ms: float | None = None
    body_trunc: str | None = Field(default=None, max_length=500)
    stream_chunks: list[str] = Field(default_factory=list)
    usage_locations: list[str] = Field(default_factory=list)
    response_fields: list[str] = Field(default_factory=list)
    response_model: str | None = None
    completion_id: str | None = None
    provider_request_id: str | None = None


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--real", action="store_true", help="guarded Groq execution; never default"
    )
    execution.add_argument(
        "--local", action="store_true", help="execute only against literal loopback"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", help="explicit execution confirmation: RUN")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--ledger", type=Path, default=Path("results/usage.sqlite3"))
    parser.add_argument("--events", type=Path, default=Path("results/events.jsonl"))
    return parser


def mode(args: argparse.Namespace) -> Literal["dry-run", "local", "real"]:
    if args.dry_run or not (args.real or args.local):
        return "dry-run"
    return "real" if args.real else "local"


def configured_url(args: argparse.Namespace, config: ProbeSettings) -> str:
    return args.base_url or (config.local_base_url if args.local else config.real_base_url)


def print_plan(plan: ProbePlan) -> None:
    print(json.dumps(Sanitizer().value(plan), ensure_ascii=True, indent=2))


def authorise_execution(args: argparse.Namespace, plan: ProbePlan, guard: RealCallGuard) -> None:
    if mode(args) == "dry-run":
        raise ProbeError("Dry-run plans cannot execute")
    local = is_loopback_url(plan.base_url)
    if (plan.mode == "local") != local:
        raise ProbeError("Local mode requires loopback; real mode requires an external provider")
    if plan.mode == "real":
        guard.check_permissions()
    if args.confirm != "RUN":
        confirmation = input("Type RUN to execute the displayed bounded probe plan: ")
        if confirmation != "RUN":
            raise ProbeError("Execution was not explicitly confirmed")


def estimate(messages: list[Message], completion_tokens: int, config: ProbeSettings) -> int:
    text = json.dumps([message.model_dump() for message in messages])
    return len(get_encoding().encode(text)) + completion_tokens + config.reservation_overhead_tokens


def tiny_messages() -> list[Message]:
    return [Message(role="user", content='Reply only with JSON {"answer":"ok"}.')]


def check_session_plan(plan: ProbePlan) -> None:
    if (
        plan.max_requests > plan.session_request_cap
        or plan.estimated_tokens > plan.session_token_cap
    ):
        raise ProbeError("Planned requests/tokens exceed the configured real session caps")


def write_json(path: Path, payload: BaseModel | dict[str, object], sanitizer: Sanitizer) -> None:
    try:
        safe = sanitizer.value(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(
            json.dumps(safe, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(path)
    except (OSError, ValueError, TypeError):
        raise ProbeError("Cannot persist sanitised probe output") from None


class ProbeSession:
    def __init__(
        self,
        args: argparse.Namespace,
        plan: ProbePlan,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.args, self.plan = args, plan
        self.clock = clock if clock is not None else SystemClock()
        self.run_id = "probe-" + uuid4().hex
        self.started_ns = self.clock.monotonic_ns()
        self.ledger: Ledger | None = None
        self.provider: GroqCompatibleProvider | None = None
        self.records: list[ProbeRecord] = []
        self.last_429_ns: int | None = None
        self.last_retry_ms: float | None = None
        self.rate_observations: list[tuple[int, float | None]] = []
        self.daily_limit_seen = False
        self.recovered_ns: int | None = None
        self.output_events: EventWriter | None = None

    async def __aenter__(self) -> "ProbeSession":
        settings = SafetySettings()
        guard = RealCallGuard(settings)  # --real is read from the actual process arguments.
        check_session_plan(self.plan)
        authorise_execution(self.args, self.plan, guard)
        # Never read a credential during planning/dry-run/local execution.
        key = os.environ.get("GROQ_API_KEY") if self.plan.mode == "real" else None
        if self.plan.mode == "real" and not key:
            raise ProbeError("GROQ_API_KEY is required for explicitly authorised real execution")
        self.ledger = Ledger(self.args.ledger, clock=self.clock)
        try:
            real_guard = (
                RealCallGuard(settings, ledger=self.ledger) if self.plan.mode == "real" else None
            )
            events = EventWriter(self.args.events, clock=self.clock)
            self.provider = GroqCompatibleProvider(
                self.plan.base_url,
                key,
                self.plan.model,
                ledger=self.ledger,
                guard=real_guard,
                events=events,
                clock=self.clock,
                session_id=self.run_id,
            )
        except Exception:
            self.ledger.close()
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        try:
            if self.provider is not None:
                await self.provider.aclose()
        finally:
            if self.ledger is not None:
                self.ledger.close()

    async def call(self, phase: str, request: LLMRequest) -> ProbeRecord:
        assert self.provider is not None
        request = request.model_copy(
            update={"metadata": CallMetadata(run_id=self.run_id, kind=phase)}
        )
        observed_utc = self.clock.utc_now().isoformat()
        common = {
            "phase": phase,
            "request_id": request.metadata.request_id,
            "call_id": request.metadata.call_id,
            "observed_at_utc": observed_utc,
            "source": "real" if self.plan.mode == "real" else "stand-in",
            "observed_against": "Groq" if self.plan.mode == "real" else "local service (not Groq)",
            "estimated_tokens": request.token_reservation,
        }
        cancelled: ProviderCancelledError | None = None
        try:
            result = await self.provider.complete(request)
            record = ProbeRecord(
                **common,
                status_code=result.status_code,
                outcome="ok",
                dispatched=True,
                usage=result.usage,
                headers=result.headers,
                header_observed_ns=result.header_snapshot.observed_ns,
                total_ms=result.total_ms,
                provider_ttft_ms=result.provider_ttft_ms,
                first_content_delta_ms=result.first_content_delta_ms,
                first_reasoning_delta_ms=result.first_reasoning_delta_ms,
                body_trunc=result.diagnostics.body_trunc,
                stream_chunks=result.diagnostics.stream_chunks,
                usage_locations=result.diagnostics.usage_locations,
                response_fields=result.diagnostics.response_fields,
                response_model=result.response_model,
                completion_id=result.completion_id,
                provider_request_id=result.provider_request_id,
            )
        except (ProviderError, ProviderCancelledError) as exc:
            if isinstance(exc, ProviderCancelledError):
                cancelled = exc
            info = exc.info
            record = ProbeRecord(
                **common,
                status_code=info.status_code,
                outcome="error",
                error_kind=info.kind,
                dispatched=bool(info.safe_context.get("dispatched")),
                usage=info.usage,
                headers=info.headers,
                retry_after_ms=info.retry_after_ms,
                total_ms=info.total_ms,
                body_trunc=info.body_trunc,
                header_observed_ns=info.safe_context.get("headers_observed_ns")
                if info.status_code is not None
                else None,
                first_content_delta_ms=info.safe_context.get("first_content_delta_ms"),
                first_reasoning_delta_ms=info.safe_context.get("first_reasoning_delta_ms"),
                provider_ttft_ms=info.safe_context.get("first_content_delta_ms"),
            )
        self.records.append(record)
        if record.status_code == 429:
            self.last_429_ns = self.clock.monotonic_ns()
            self.last_retry_ms = record.retry_after_ms
            self.rate_observations.append((self.last_429_ns, self.last_retry_ms))
            self.recovered_ns = None
            self.daily_limit_seen |= record.error_kind == "rate_limit_day"
        elif record.status_code is not None and 200 <= record.status_code < 300:
            if self.last_429_ns is not None and self.recovered_ns is None:
                self.recovered_ns = self.clock.monotonic_ns()
        if self.output_events is not None:
            self.output_events.write(
                EventContext(rid=record.request_id, run=self.run_id, arrival_ns=self.started_ns),
                "probe_attempt",
                **record.model_dump(mode="json"),
            )
        if cancelled is not None:
            raise cancelled
        return record

    async def cooldown(self, config: ProbeSettings, sleep: Sleep = asyncio.sleep) -> None:
        if self.last_429_ns is None:
            return
        if self.daily_limit_seen:
            raise ProbeError("Daily quota evidence: abort all further provider calls")
        until = max(
            observed
            + int(
                (retry / 1000 if retry is not None else config.recovery_interval_s) * 1_000_000_000
            )
            for observed, retry in self.rate_observations
        )
        wait = max(0.0, (until - self.clock.monotonic_ns()) / 1_000_000_000)
        if wait > config.max_recovery_wait_s:
            raise ProbeError(
                "Retry-After exceeds the bounded wait; no early follow-up will be sent"
            )
        if wait:
            await sleep(wait)
