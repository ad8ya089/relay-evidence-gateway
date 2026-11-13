"""Progressive measured rate points; development observations are always S_dev."""

import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from loadgen.runner import commit_id, execute
from loadgen.scenarios import Scenario, catalog
from loadgen.settings import load_settings
from loadgen.workload import questions
from relay.obs.clock import SystemClock
from scripts.aggregate import capacity_point, capacity_result
from scripts.build_results_csv import build
from standin.profile import load_profile


async def discover(
    *,
    root: Path = Path("results"),
    fixture_only: bool = False,
    tiny: bool = False,
    on_port: Callable[[int], None] | None = None,
) -> dict[str, object]:
    settings, clock = load_settings(), SystemClock()
    if tiny and not fixture_only:
        raise ValueError("--tiny requires --fixture-only; it cannot measure official S")
    report: dict[str, object] = {
        "capacity_id": f"capacity-{uuid4().hex}",
        "git_commit": commit_id(),
        "measured_at": clock.utc_now().isoformat(timespec="milliseconds"),
        "fixture_only": fixture_only,
        "points": [],
        "completion_status": "preparing",
        "value": None,
    }
    template = catalog()[1]["development_smoke" if fixture_only else "steady"]
    try:
        try:
            profile = load_profile("scaled")
        except ValueError:
            profile = load_profile("development")
        # Fixture-only experiments never use an account-calibrated profile.
        if fixture_only:
            profile = load_profile("development")
        report.update(
            kind="S" if profile.name == "scaled" else "S_dev",
            profile=profile.name,
            profile_snapshot=profile.model_dump(),
            profile_provenance=profile.model_dump(),
            seed=template.seed,
            deadline_ms=template.deadline_ms,
            setting=template.setting,
            capacity_settings=settings.capacity.model_dump(),
        )
        questions(template.workload, settings, fixture_only=fixture_only)
        rates = (
            settings.capacity.tiny_rates_per_minute if tiny else settings.capacity.rates_per_minute
        )
        duration = settings.capacity.tiny_duration_s if tiny else settings.capacity.duration_s
        for index, rate in enumerate(rates):
            point_scenario = Scenario.model_validate(
                template.model_dump()
                | {
                    "name": f"{'development_' if fixture_only else ''}capacity_{index}",
                    "label": f"{report['kind']} rate point",
                    "profile": profile.name,
                    "duration_s": duration,
                    "expected_requests": f"{rate} arrivals/minute for {duration}s",
                    "arrivals": {
                        "kind": "constant",
                        "segments": [{"duration_s": duration, "rate_per_minute": rate}],
                    },
                    "cache": False,
                    "sampling": "uniform",
                    "retry_policy": "bounded",
                    "admin_events": [],
                    "policy_overrides": {},
                    "daily_budget_fraction": None,
                    "faults": settings.capacity.faults,
                }
            )
            run = await execute(point_scenario, root=root, on_port=on_port)
            if run.manifest.completion_status != "complete":
                raise RuntimeError(f"Capacity point did not complete: {run.manifest.reason}")
            rows = build(
                [run.manifest_path.parent / "client_rows.jsonl"],
                [run.manifest_path.parent / "events.jsonl"],
                [run.manifest_path],
            )
            point = capacity_point(
                rows,
                rate,
                template.deadline_ms,
                settings.capacity.minimum_success,
                settings.capacity.minimum_samples,
            )
            point["run_id"], point["manifest"] = run.manifest.run_id, str(run.manifest_path)
            report["points"].append(point)
        report.update(capacity_result(report["points"]))
        chosen = next(
            (p for p in report["points"] if p["rate_per_minute"] == report["value"]), None
        )
        report["mean_counted_tokens_per_answer"] = (
            chosen["metrics"]["mean_counted_tokens_per_answer"] if chosen else None
        )
        report["token_estimate_basis"] = (
            "provider-backed successful logical requests at selected passing rate; not calibration"
        )
        report["completion_status"] = "complete"
    except (ValueError, OSError, RuntimeError) as exc:
        report["completion_status"], report["reason"] = (
            "blocked" if not report["points"] else "incomplete",
            str(exc)[:500],
        )
    directory = root / "capacity_reports"
    directory.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2)
    (directory / f"{report['capacity_id']}.json").write_text(content, encoding="utf-8")
    temporary = root / "capacity_report.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(root / "capacity_report.json")
    return report
