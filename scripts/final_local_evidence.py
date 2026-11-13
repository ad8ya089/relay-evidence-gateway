"""New controlled development runs; no provider credentials or real network endpoints."""

import asyncio
import copy
import ctypes
import json
import sys
from pathlib import Path

import yaml

from loadgen.runner import execute
from loadgen.scenarios import catalog


async def run() -> None:
    root = Path("results/final_completion/experiments")
    root.mkdir(exist_ok=False)
    data = yaml.safe_load(
        Path("results/evidence_completion/experiments/scenarios.yaml").read_text()
    )
    selected = [
        "development_minute_limit_halved",
        "development_retry_naive",
        "development_retry_bounded",
    ]
    data["order"] = selected
    data["scenarios"] = {n: copy.deepcopy(data["scenarios"][n]) for n in selected}
    minute = data["scenarios"][selected[0]]
    minute["arrivals"]["segments"][0]["rate_per_minute"] = 30
    minute["admin_events"] = [
        {"offset_s": 0, "action": "limits", "values": {"tpm": 30000}},
        {"offset_s": 8, "action": "halve_minute_limits", "factor": 0.5},
    ]
    config = root / "scenarios.yaml"
    config.write_text(yaml.safe_dump(data))
    timer_requested = sys.platform == "win32" and ctypes.windll.winmm.timeBeginPeriod(1) == 0
    (root / "timer_metadata.json").write_text(
        json.dumps(
            {
                "requested_timer_period_ms": 1,
                "accepted": timer_requested,
                "clock": "time.monotonic_ns unchanged; Windows clock quantization may remain",
                "pair_conditions": (
                    "same seed, workload, schedule, profile, faults and limits; retry "
                    "policy differs"
                ),
            },
            indent=2,
        )
    )
    try:
        _, scenarios = catalog(config)
        for name in selected:
            result = await execute(scenarios[name], root=root)
            print(
                json.dumps(
                    {
                        "run_id": result.manifest.run_id,
                        "scenario": name,
                        "status": result.manifest.completion_status,
                    }
                ),
                flush=True,
            )
    finally:
        if timer_requested:
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    asyncio.run(run())
