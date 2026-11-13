"""Explicit local experiments; default commands never discover credentials or call Groq."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from loadgen.capacity import discover
from loadgen.real_burst import run as real_burst
from loadgen.runner import execute
from loadgen.scenarios import catalog
from relay.obs.events import Sanitizer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run a configured local scenario; official prerequisites are mandatory"
    )
    run.add_argument("--scenario", required=True)
    run.add_argument("--capacity-report", type=Path, default=Path("results/capacity_report.json"))
    run.add_argument("--config", type=Path, default=Path("config/scenarios.yaml"))
    all_scenarios = commands.add_parser(
        "scenarios", help="run configured official order using fresh isolated services per run"
    )
    all_scenarios.add_argument(
        "--capacity-report", type=Path, default=Path("results/capacity_report.json")
    )
    all_scenarios.add_argument("--config", type=Path, default=Path("config/scenarios.yaml"))
    capacity = commands.add_parser(
        "capacity", help="measure progressive local rate points (S or explicitly S_dev)"
    )
    capacity.add_argument("--fixture-only", action="store_true")
    capacity.add_argument("--tiny", action="store_true")
    real = commands.add_parser(
        "real-burst", help="future opt-in against an existing guarded real Relay; never default"
    )
    real.add_argument("--n", type=int, default=6)
    real.add_argument("--real", action="store_true")
    real.add_argument("--confirm", choices=["RUN"])
    real.add_argument("--relay-url", required=True)
    real.add_argument("--model", required=True)
    real.add_argument("--policy", type=Path, default=Path("config/policy.yaml"))
    for command in (run, all_scenarios, capacity, real):
        command.add_argument("--output-root", type=Path, default=Path("results"))
    return result


async def invoke(args: argparse.Namespace) -> int:
    if args.command == "capacity":
        report = await discover(
            root=args.output_root, fixture_only=args.fixture_only, tiny=args.tiny
        )
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in (
                        "kind",
                        "value",
                        "censored",
                        "completion_status",
                        "reason",
                        "generator_warning_at_selected_rate",
                    )
                },
                indent=2,
            )
        )
        return 0 if report["completion_status"] == "complete" else 2
    if args.command == "real-burst":
        rows = await real_burst(
            url=args.relay_url,
            n=args.n,
            real=args.real,
            allow_real=os.environ.get("RELAY_ALLOW_REAL"),
            confirmation=args.confirm,
            model=args.model,
            root=args.output_root,
            policy_path=args.policy,
        )
        print(f"Recorded {len(rows)} logical real-burst results")
        return 0
    order, scenarios = catalog(args.config)
    names = [args.scenario] if args.command == "run" else order
    failed = False
    for name in names:
        if name not in scenarios:
            raise ValueError("Unknown scenario; see config/scenarios.yaml")
        report = await execute(
            scenarios[name], root=args.output_root, capacity_path=args.capacity_report
        )
        print(
            json.dumps(
                {
                    "run_id": report.manifest.run_id,
                    "scenario": name,
                    "completion_status": report.manifest.completion_status,
                    "reason": report.manifest.reason,
                    "manifest": str(report.manifest_path),
                    "generator_lag": report.manifest.details.get("generator_lag"),
                },
                indent=2,
            )
        )
        failed |= report.manifest.completion_status != "complete"
    return 2 if failed else 0


def main() -> int:
    args = parser().parse_args()
    try:
        return asyncio.run(invoke(args))
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        print(Sanitizer().text(str(exc), body=True))
        return 2
    except KeyboardInterrupt:
        print("Interrupted; owned child cleanup completed and raw records were retained")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
