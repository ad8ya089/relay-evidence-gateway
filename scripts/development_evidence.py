"""Explicitly uncalibrated local experiments using candidate text, never gold prompts."""

import asyncio
import json
from pathlib import Path

import yaml

from eval.dataset import load_key, load_questions
from loadgen.runner import execute
from loadgen.scenarios import catalog

ROOT = Path("results/evidence_completion")


async def run() -> None:
    root = ROOT / "experiments"
    root.mkdir(exist_ok=False)
    keys = load_key()
    workload = root / "workload.jsonl"
    workload.write_text(
        "\n".join(
            json.dumps(
                {
                    "qid": q.qid,
                    "question": q.question,
                    "paper": q.paper,
                    "answerable": keys[q.qid].answerable,
                    "type": "explain" if keys[q.qid].answerable else "absent",
                }
            )
            for q in load_questions()
        )
        + "\n"
    )
    # Answerability is scheduler metadata only; dispatch sends only q.question.
    defaults = dict(
        seed=20261114,
        profile="development",
        setting="A",
        deadline_ms=8000,
        cache=False,
        sampling="uniform",
        retry_policy="bounded",
        faults={
            "server_error_prob": 0,
            "timeout_prob": 0,
            "disconnect_prob": 0,
            "slow_success_prob": 0,
            "verify_fail_prob": 0,
        },
        admin_events=[],
        policy_overrides={},
        fixture_only=True,
        workload=str(workload),
        align_policy_limits=True,
        duration_s=24,
        expected_requests="chosen local development schedule; not calibrated capacity",
    )
    scenarios = {}
    for name, rate in [
        ("steady", 30),
        ("burst", 120),
        ("sustained_overload", 240),
        ("minute_limit_halved", 60),
        ("daily_exhausted", 60),
        ("retry_naive", 60),
        ("retry_bounded", 60),
        ("burst_cache_on", 120),
    ]:
        data = dict(
            label="Uncalibrated development " + name,
            arrivals={
                "kind": "constant",
                "segments": [{"duration_s": 24, "rate_per_minute": rate}],
            },
        )
        if name in {"burst", "burst_cache_on"}:
            data["arrivals"] = {
                "kind": "burst",
                "segments": [
                    {"duration_s": 8, "rate_per_minute": 30},
                    {"duration_s": 8, "rate_per_minute": 300},
                    {"duration_s": 8, "rate_per_minute": 30},
                ],
            }
        if name == "minute_limit_halved":
            data["admin_events"] = [{"offset_s": 8, "action": "halve_minute_limits", "factor": 0.5}]
        if name == "daily_exhausted":
            data["admin_events"] = [{"offset_s": 8, "action": "limits", "values": {"tpd": 1}}]
        if name.startswith("retry_"):
            data.update(
                retry_policy="naive_s7a" if name == "retry_naive" else "bounded",
                faults=defaults["faults"] | {"slow_success_prob": 0.5, "slow_factor": 8},
            )
        if name == "burst_cache_on":
            data.update(cache=True, sampling="zipf")
        scenarios["development_" + name] = data
    config = root / "scenarios.yaml"
    config.write_text(
        yaml.safe_dump({"defaults": defaults, "order": list(scenarios), "scenarios": scenarios})
    )
    order, parsed = catalog(config)
    for name in order:
        report = await execute(parsed[name], root=root)
        print(
            json.dumps(
                {
                    "run_id": report.manifest.run_id,
                    "scenario": name,
                    "status": report.manifest.completion_status,
                    "reason": report.manifest.reason,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(run())
