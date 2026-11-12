"""Bounded RPM/TPM/concurrency/recovery experiments; dry-run is the default."""

import argparse
import asyncio
import math
from pathlib import Path

from relay.control.ledger import Ledger, LedgerError
from relay.obs.events import EventWriter
from relay.providers.base import LLMRequest, Message, ResponseFormat
from relay.providers.guard import RealCallDenied
from relay.retrieval.tokenizer import get_encoding
from relay.settings import SafetySettings
from scripts.probe_common import (
    AccountLimits,
    PhasePlan,
    ProbeError,
    ProbePlan,
    ProbeSession,
    ProbeSettings,
    Sleep,
    common_parser,
    configured_url,
    estimate,
    mode,
    print_plan,
    probe_settings,
    tiny_messages,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = common_parser(__doc__ or "Bounded limit probes")
    result.add_argument("--output", type=Path, default=Path("results/limit_probe.jsonl"))
    result.add_argument("--summary", type=Path, default=Path("results/limit_probe_summary.json"))
    for name in (
        "rpm-requests",
        "tpm-requests",
        "tpm-input-tokens",
        "concurrency-requests",
        "recovery-requests",
        "max-completion-tokens",
        "remaining-day-requests",
        "remaining-day-tokens",
    ):
        result.add_argument("--" + name, type=int)
    result.add_argument(
        "--rpm-rates", help="comma-separated offered rates/minute, not provider limits"
    )
    for name in ("rpm", "tpm", "rpd", "tpd"):
        result.add_argument("--account-" + name, type=int)
    return result


def configured(args: argparse.Namespace, config: ProbeSettings) -> ProbeSettings:
    fields = config.model_dump()
    for name in (
        "rpm_requests",
        "tpm_requests",
        "tpm_input_tokens",
        "concurrency_requests",
        "recovery_requests",
        "remaining_day_requests",
        "remaining_day_tokens",
    ):
        if (value := getattr(args, name, None)) is not None:
            fields[name] = value
    if args.model:
        fields["model"] = args.model
    if args.max_completion_tokens is not None:
        fields["limit_max_completion_tokens"] = args.max_completion_tokens
    if args.rpm_rates:
        fields["rpm_rates"] = [float(value) for value in args.rpm_rates.split(",")]
    account = config.account_limits.model_dump()
    for name in account:
        if (value := getattr(args, "account_" + name)) is not None:
            account[name] = value
    fields["account_limits"] = AccountLimits.model_validate(account)
    updated = ProbeSettings.model_validate(fields)
    if any(
        not math.isfinite(rate) or rate <= 0 for rate in updated.rpm_rates
    ) or updated.rpm_rates != sorted(updated.rpm_rates):
        raise ProbeError("RPM offered rates must be finite, positive and increasing")
    return updated


def phase_requests(config: ProbeSettings) -> tuple[LLMRequest, LLMRequest]:
    encoding = get_encoding()
    padding = (
        'Rate-limit probe. Reply with JSON {"answer":"ok"}. ' + " filler" * config.tpm_input_tokens
    )
    long_prompt = encoding.decode(encoding.encode(padding)[: config.tpm_input_tokens])
    tiny, large = tiny_messages(), [Message(role="user", content=long_prompt)]

    def make(messages: list[Message]) -> LLMRequest:
        return LLMRequest(
            model=config.model,
            messages=messages,
            max_completion_tokens=config.limit_max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
            response_format=ResponseFormat(type="json_object"),
            token_reservation=estimate(messages, config.limit_max_completion_tokens, config),
            capture_response=True,
        )

    return make(tiny), make(large)


def build_plan(
    args: argparse.Namespace, config: ProbeSettings
) -> tuple[ProbePlan, LLMRequest, LLMRequest]:
    tiny, large = phase_requests(config)
    assert tiny.token_reservation is not None and large.token_reservation is not None
    tpm_budget = config.tpm_requests * large.token_reservation
    if tpm_budget > 10000:  # Explicit per-phase ceiling required by P2.
        raise ProbeError("TPM phase exceeds its hard 10K estimated-token ceiling")
    phases = [
        PhasePlan(
            name="rpm",
            max_requests=config.rpm_requests,
            estimated_tokens=config.rpm_requests * tiny.token_reservation,
            offered_rates_per_minute=config.rpm_rates,
        ),
        PhasePlan(
            name="tpm",
            max_requests=config.tpm_requests,
            concurrency=config.tpm_requests,
            estimated_tokens=tpm_budget,
        ),
        PhasePlan(
            name="concurrency",
            max_requests=config.concurrency_requests,
            concurrency=config.concurrency_requests,
            estimated_tokens=config.concurrency_requests * tiny.token_reservation,
        ),
        PhasePlan(
            name="recovery",
            max_requests=config.recovery_requests,
            estimated_tokens=config.recovery_requests * tiny.token_reservation,
        ),
    ]
    safety = SafetySettings()
    return (
        ProbePlan(
            mode=mode(args),
            model=config.model,
            base_url=configured_url(args, config),
            phases=phases,
            max_requests=sum(p.max_requests for p in phases),
            estimated_tokens=sum(p.estimated_tokens for p in phases),
            account_limits=config.account_limits,
            remaining_day_requests=config.remaining_day_requests,
            remaining_day_tokens=config.remaining_day_tokens,
            daily_safety_fraction=config.daily_safety_fraction,
            session_request_cap=safety.real_request_cap,
            session_token_cap=safety.real_token_cap,
        ),
        tiny,
        large,
    )


def check_daily_headroom(config: ProbeSettings, plan: ProbePlan, ledger: Ledger) -> None:
    limits = config.account_limits
    if any(value is None for value in limits.model_dump().values()):
        raise ProbeError(
            "Real limit probes require account RPM/TPM/RPD/TPD evidence; none is invented"
        )
    if config.remaining_day_requests is None or config.remaining_day_tokens is None:
        raise ProbeError(
            "Supply verified remaining-day requests/tokens before a real limit experiment"
        )
    assert limits.rpd is not None and limits.tpd is not None
    if config.remaining_day_requests > limits.rpd or config.remaining_day_tokens > limits.tpd:
        raise ProbeError("Remaining-day inputs cannot exceed the supplied daily limits")
    local = ledger.daily_usage(plan.model)
    if local.unreserved_unknown_count or local.inconsistent_usage_count:
        raise ProbeError("Unbounded or inconsistent historical usage requires review")
    available_requests = min(config.remaining_day_requests, limits.rpd - local.attempts)
    available_tokens = min(config.remaining_day_tokens, limits.tpd - local.guard_charged_tokens)
    if (
        plan.max_requests > available_requests * config.daily_safety_fraction
        or plan.estimated_tokens > available_tokens * config.daily_safety_fraction
    ):
        raise ProbeError("Plan exceeds conservative daily headroom; daily exhaustion is prohibited")


def ensure_continuable(session: ProbeSession) -> None:
    if session.daily_limit_seen:
        raise ProbeError("Daily quota evidence: abort all further provider calls")
    if any(r.error_kind in {"budget_exceeded", "persistence"} for r in session.records):
        raise ProbeError("Stopped by session cap or persistence protection")


async def rpm_phase(
    session: ProbeSession, req: LLMRequest, config: ProbeSettings, sleep: Sleep
) -> None:
    next_due = session.clock.monotonic_ns()
    for index in range(config.rpm_requests):
        remaining = (next_due - session.clock.monotonic_ns()) / 1_000_000_000
        if remaining > 0:
            await sleep(remaining)
        rate = config.rpm_rates[
            min(index * len(config.rpm_rates) // config.rpm_requests, len(config.rpm_rates) - 1)
        ]
        next_due = session.clock.monotonic_ns() + int(60 / rate * 1_000_000_000)
        record = await session.call("rpm", req)
        ensure_continuable(session)
        if record.status_code == 429:
            break


async def concurrent_phase(session: ProbeSession, req: LLMRequest, phase: str, count: int) -> None:
    # Each is a distinct planned experiment, not a retry of a failed logical request.
    outcomes = await asyncio.gather(
        *[session.call(phase, req) for _ in range(count)], return_exceptions=True
    )
    if any(isinstance(outcome, BaseException) for outcome in outcomes):
        raise ProbeError("Concurrent phase did not complete; inspect persisted attempts")
    ensure_continuable(session)


async def recovery_phase(
    session: ProbeSession, req: LLMRequest, config: ProbeSettings, sleep: Sleep
) -> None:
    if session.last_429_ns is None or session.recovered_ns is not None:
        return
    for _ in range(config.recovery_requests):
        await session.cooldown(config, sleep)
        record = await session.call("recovery", req)
        ensure_continuable(session)
        if record.status_code is not None and 200 <= record.status_code < 300:
            break
        if record.status_code != 429:
            await sleep(config.recovery_interval_s)


async def run(
    args: argparse.Namespace, config: ProbeSettings | None = None, *, sleep: Sleep = asyncio.sleep
) -> int:
    config = configured(args, config if config is not None else probe_settings())
    plan, tiny, large = build_plan(args, config)
    print_plan(plan)
    if plan.mode == "dry-run":
        print(
            "Dry run: no provider client, ledger or network opened; all hosted limits unverified."
        )
        return 0
    if plan.mode == "real":
        # Check available budget before the explicit confirmation in ProbeSession.
        with Ledger(args.ledger) as ledger:
            check_daily_headroom(config, plan, ledger)
    async with ProbeSession(args, plan) as session:
        assert session.provider is not None
        session.output_events = EventWriter(
            args.output, clock=session.clock, sanitizer=session.provider.sanitizer
        )
        aborted: str | None = None
        try:
            await rpm_phase(session, tiny, config, sleep)
            await session.cooldown(config, sleep)
            await concurrent_phase(session, large, "tpm", config.tpm_requests)
            await session.cooldown(config, sleep)
            await concurrent_phase(session, tiny, "concurrency", config.concurrency_requests)
            await recovery_phase(session, tiny, config, sleep)
        except ProbeError as exc:
            aborted = str(exc)
        recovery_ms = None
        if session.recovered_ns is not None and session.last_429_ns is not None:
            recovery_ms = (session.recovered_ns - session.last_429_ns) / 1_000_000
        write_json(
            args.summary,
            {
                "plan": plan,
                "run_id": session.run_id,
                "attempts": len(session.records),
                "observed_recovery_ms": recovery_ms,
                "reported_retry_after_ms": session.last_retry_ms,
                "recovery_definition": (
                    "First observed HTTP 2xx after the latest 429, separate from JSON validation"
                ),
                "aborted": aborted,
                "source": "real" if plan.mode == "real" else "stand-in",
                "unverified": [
                    "Actual provider quota consumption of failures",
                    "Published concurrency limits cannot be inferred from rate headers",
                ],
            },
            session.provider.sanitizer,
        )
        print(
            f"Captured {len(session.records)} local attempted-request records; source={plan.mode}."
        )
        if aborted:
            raise ProbeError(aborted)
    return 0


def main() -> int:
    try:
        return asyncio.run(run(parser().parse_args()))
    except (ProbeError, RealCallDenied) as exc:
        print(f"Limit probe refused: {exc}")
        return 2
    except (ValueError, OSError, LedgerError):
        print("Limit probe refused: invalid configuration or unavailable persistence")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
