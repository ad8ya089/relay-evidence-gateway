"""Inspect completion shapes; dry-run by default. Real execution is separately authorised."""

import argparse
import asyncio
from pathlib import Path

from pydantic import BaseModel

from relay.providers.base import JsonSchemaDefinition, LLMRequest, ResponseFormat
from relay.providers.guard import RealCallDenied
from relay.settings import SafetySettings
from scripts.probe_common import (
    PhasePlan,
    ProbeError,
    ProbePlan,
    ProbeSession,
    ProbeSettings,
    common_parser,
    configured_url,
    estimate,
    mode,
    print_plan,
    probe_settings,
    tiny_messages,
    write_json,
)


class ProbeAnswer(BaseModel):
    answer: str


def parser() -> argparse.ArgumentParser:
    result = common_parser(__doc__ or "Probe response shapes")
    result.add_argument("--experiment-stream-schema", action="store_true")
    result.add_argument("--output", type=Path, default=Path("results/probe_shapes.json"))
    return result


def requests(config: ProbeSettings, experiment: bool) -> dict[str, LLMRequest]:
    messages = tiny_messages()
    schema = ResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(
            name="probe_answer", schema_spec=ProbeAnswer.model_json_schema()
        ),
    )
    budget = estimate(messages, config.shapes_max_completion_tokens, config)
    common: dict[str, object] = {
        "messages": messages,
        "model": config.model,
        "max_completion_tokens": config.shapes_max_completion_tokens,
        "reasoning_effort": config.reasoning_effort,
        "token_reservation": budget,
        "validation_model": ProbeAnswer,
        "capture_response": True,
    }
    variants = {
        "nonstream_json_schema": LLMRequest.model_validate(common | {"response_format": schema}),
        "stream_json_object": LLMRequest.model_validate(
            common
            | {
                "stream": True,
                "response_format": ResponseFormat(type="json_object"),
            }
        ),
    }
    if experiment:
        variants["schema_stream_experiment"] = LLMRequest.model_validate(
            common
            | {
                "stream": True,
                "response_format": schema,
                "schema_stream_policy": "experiment",
            }
        )
    return variants


def build_plan(
    args: argparse.Namespace, config: ProbeSettings
) -> tuple[ProbePlan, dict[str, LLMRequest]]:
    if args.model:
        config = config.model_copy(update={"model": args.model})
    variants = requests(config, args.experiment_stream_schema)
    phases = [
        PhasePlan(name=name, max_requests=1, estimated_tokens=req.token_reservation or 0)
        for name, req in variants.items()
    ]
    safety = SafetySettings()
    return ProbePlan(
        mode=mode(args),
        model=config.model,
        base_url=configured_url(args, config),
        phases=phases,
        max_requests=len(phases),
        estimated_tokens=sum(p.estimated_tokens for p in phases),
        account_limits=config.account_limits,
        session_request_cap=safety.real_request_cap,
        session_token_cap=safety.real_token_cap,
    ), variants


async def run(args: argparse.Namespace, config: ProbeSettings | None = None) -> int:
    config = config if config is not None else probe_settings()
    plan, variants = build_plan(args, config)
    print_plan(plan)
    if plan.mode == "dry-run":
        print("Dry run: no provider client, ledger or network opened; hosted behaviour unverified.")
        return 0
    async with ProbeSession(args, plan) as session:
        assert session.provider is not None
        for name, req in variants.items():
            await session.cooldown(config)
            record = await session.call(name, req)
            write_json(
                args.output,
                {
                    "plan": plan,
                    "run_id": session.run_id,
                    "documented": {
                        "checked_on": "2026-11-12",
                        "structured_outputs_streaming": "unsupported per documentation",
                        "source": "https://console.groq.com/docs/structured-outputs",
                    },
                    "observed": session.records,
                    "unverified": [
                        "Hosted Groq behaviours are unverified in local runs",
                        "An arbitrary 400 does not identify which parameter is unsupported",
                        "Account-specific limits and uncaptured model/parameter combinations",
                    ],
                },
                session.provider.sanitizer,
            )
            print(
                f"{name}: HTTP {record.status_code}, {record.outcome}, "
                f"{record.error_kind or 'completed'}"
            )
            if record.error_kind in {"budget_exceeded", "persistence"}:
                raise ProbeError("Execution stopped by budget or persistence protection")
    return 0


def main() -> int:
    try:
        return asyncio.run(run(parser().parse_args()))
    except (ProbeError, RealCallDenied) as exc:
        print(f"Probe refused: {exc}")
        return 2
    except (ValueError, OSError):
        print("Probe refused: invalid configuration or unavailable output path")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
