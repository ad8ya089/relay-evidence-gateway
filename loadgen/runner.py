"""Raw experiment execution; final result figures live exclusively in aggregate.py."""

import asyncio
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import yaml
from pydantic import Field

from loadgen.arrivals import plan
from loadgen.client import LoadClient, dispatch
from loadgen.models import ClientRecord, RunContext
from loadgen.scenarios import AdminEvent, Scenario, prerequisite
from loadgen.service import ServiceSpec
from loadgen.services import OwnedService
from loadgen.settings import LoadSettings, Model, load_settings
from loadgen.workload import PrerequisiteError
from relay.control.settings import Policy
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriter
from scripts.aggregate import lag_summary
from scripts.build_results_csv import json_rows
from standin.profile import Profile


class RunManifest(Model):
    run_id: str
    scenario: str
    source: Literal["stand-in", "real"] = "stand-in"
    fixture_only: bool
    completion_status: Literal["preparing", "blocked", "running", "complete", "incomplete"] = (
        "preparing"
    )
    reason: str | None = None
    start_time: str
    git_commit: str
    scenario_snapshot: dict[str, object]
    loadgen_snapshot: dict[str, object]
    details: dict[str, object] = Field(default_factory=dict)


class RunReport(Model):
    manifest_path: Path
    manifest: RunManifest
    rows: list[ClientRecord] = Field(default_factory=list)


def persist(manifest: RunManifest, directory: Path, root: Path) -> None:
    # Details are also exposed at the top level for cross-tool manifest joins.
    data = manifest.model_dump(mode="json") | manifest.details
    for path in (directory / "run_manifest.json", root / "run_manifest.json"):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)


def commit_id() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def configured_profile(profile: Profile, scenario: Scenario, history: int) -> Profile:
    data = profile.model_dump()
    data["seed"] = {"value": scenario.seed, "source": "chosen", "note": "Scenario seed"}
    data["diagnostic_history"] = {
        "value": history,
        "source": "chosen",
        "note": "Bounded full-run sampling audit",
    }
    for name, value in scenario.faults.items():
        if name not in data["faults"]:
            raise ValueError("Unknown scenario fault parameter")
        data["faults"][name] = {
            "value": value,
            "source": "chosen",
            "note": "Scenario fault override",
        }
    return Profile.model_validate(data).runnable()


def configured_policy(scenario: Scenario, profile: Profile) -> Policy:
    data = yaml.safe_load(Path("config/policy.yaml").read_text())
    data.update(scenario.policy_overrides)
    data["deadline_ms"] = scenario.deadline_ms
    data["max_queue_wait_ms"] = min(data["max_queue_wait_ms"], scenario.deadline_ms)
    data["cache"]["enabled"] = scenario.cache
    data["retry"]["policy"] = scenario.retry_policy
    data["retry"]["seed"] = scenario.seed
    if scenario.align_policy_limits:
        data["limits"] = {m: limits.numbers() for m, limits in profile.models.items()}
    return Policy.model_validate(data)


async def quiescent(
    http: httpx.AsyncClient, standin: str, relay: str, clock: Clock, settings: LoadSettings
) -> tuple[dict[str, object], dict[str, object]]:
    end = clock.monotonic_ns() + int(settings.quiescence_timeout_s * 1e9)
    while True:
        a = await http.get(standin + "/admin/state")
        b = await http.get(relay + "/state")
        a.raise_for_status()
        b.raise_for_status()
        sa, sb = a.json(), b.json()
        if (
            sa["inflight"]
            == sb["active_logical_requests"]
            == sb["queue_length"]
            == sb["provider"]["inflight"]
            == 0
        ):
            return sa, sb
        if clock.monotonic_ns() >= end:
            raise TimeoutError("Services did not reach quiescence; run is incomplete")
        await asyncio.sleep(settings.health_poll_s)


async def administer(
    events: list[AdminEvent],
    base_ns: int,
    http: httpx.AsyncClient,
    standin: str,
    relay: str,
    clock: Clock,
    settings: LoadSettings,
    writer: EventWriter,
    run_id: str,
) -> list[dict[str, object]]:
    records = []
    for event in events:
        scheduled = base_ns + int(event.offset_s * 1e9)
        while (remaining := scheduled - clock.monotonic_ns()) > 0:
            await asyncio.sleep(remaining / 1e9)
        record = {
            "scheduled_monotonic": scheduled,
            "action": event.action,
            "actual_monotonic": clock.monotonic_ns(),
            "status": "incomplete",
            "response_status": None,
            "applied_values": None,
        }
        try:
            if event.action in {"profile", "reset", "faults"}:
                state, _ = await quiescent(http, standin, relay, clock, settings)
                records.append({"action": "pre_reset_sampling_snapshot", "state": state})
            else:
                response = await http.get(standin + "/admin/state")
                response.raise_for_status()
                state = response.json()
            payload = event.values.copy()
            endpoint = event.action
            if event.action == "halve_minute_limits":
                limits = state["models"][state["model"]]["limits"]
                payload = {k: max(1, math.floor(limits[k] * event.factor)) for k in ("rpm", "tpm")}
                endpoint = "limits"
            elif event.action == "faults":
                data = state["profile"]
                for key, value in payload.items():
                    data["faults"][key] = {
                        "value": value,
                        "source": "chosen",
                        "note": "Scheduled scenario fault change",
                    }
                payload, endpoint = (
                    {"profile": Profile.model_validate(data).model_dump()},
                    "profile",
                )
            record["actual_monotonic"] = clock.monotonic_ns()
            response = await http.post(standin + "/admin/" + endpoint, json=payload)
            record["response_status"] = response.status_code
            response.raise_for_status()
            record.update(status="applied", applied_values=response.json())
        except (ValueError, TimeoutError, httpx.HTTPError) as exc:
            record["reason"] = type(exc).__name__
        records.append(record)
        writer.write(
            EventContext(rid=f"{run_id}-admin", run=run_id, arrival_ns=base_ns),
            "admin_event",
            **record,
        )
        if record["status"] != "applied":
            raise RuntimeError("Scheduled admin event failed; inspect admin events")
    return records


async def execute(
    scenario: Scenario,
    *,
    root: Path = Path("results"),
    capacity_path: Path = Path("results/capacity_report.json"),
    settings: LoadSettings | None = None,
    clock: Clock | None = None,
    on_port: Callable[[int], None] | None = None,
) -> RunReport:
    clock, settings = clock or SystemClock(), settings or load_settings()
    run_id = f"p6-{uuid4().hex}"
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = clock.monotonic_ns()
    manifest = RunManifest(
        run_id=run_id,
        scenario=scenario.name,
        fixture_only=scenario.fixture_only,
        start_time=clock.utc_now().isoformat(timespec="milliseconds"),
        git_commit=commit_id(),
        scenario_snapshot=scenario.model_dump(mode="json"),
        loadgen_snapshot=settings.model_dump(),
    )
    report = RunReport(manifest_path=directory / "run_manifest.json", manifest=manifest)
    persist(manifest, directory, root)
    children: list[OwnedService] = []
    client = None
    admin = None
    event_path = root / "events.jsonl"
    admin_writer = EventWriter(directory / "admin_events.jsonl", clock=clock)
    try:
        workload, original, capacity = prerequisite(scenario, settings, capacity_path)
        profile = configured_profile(
            original, scenario, settings.max_requests * settings.diagnostic_calls_per_request
        )
        plans = plan(
            scenario.arrivals,
            list(workload),
            run_id=run_id,
            seed=scenario.seed,
            setting=scenario.setting,
            sampling=scenario.sampling,
            maximum=settings.max_requests,
            exponent=settings.zipf_exponent,
            capacity=capacity["value"] if capacity else None,
        )
        if scenario.daily_budget_fraction:
            data = profile.model_dump()
            tokens = math.ceil(
                len(plans)
                * capacity["mean_counted_tokens_per_answer"]
                * scenario.daily_budget_fraction
            )
            data["models"][profile.model]["tpd"] = {
                "value": max(1, tokens),
                "source": "chosen",
                "note": "S5 budget: measured counted tokens/answer * planned requests * fraction",
            }
            profile = Profile.model_validate(data).runnable()
        policy = configured_policy(scenario, profile)
        policy_path = directory / "policy.yaml"
        policy_path.write_text(yaml.safe_dump(policy.model_dump()), encoding="utf-8")
        manifest.details.update(
            profile=profile.name,
            model=profile.model,
            profile_snapshot=profile.model_dump(),
            profile_provenance=original.model_dump(),
            actual_limits={m: q.numbers() for m, q in profile.models.items()},
            capacity=profile.capacity.model_dump(),
            alpha=profile.alpha.model_dump(),
            seed=scenario.seed,
            policy_snapshot=policy.model_dump(),
            planned_requests=[p.model_dump() for p in plans],
            question_ids=list(workload),
            sampling=scenario.sampling,
            zipf_exponent=settings.zipf_exponent,
            scheduled_duration_s=scenario.duration_s,
            offered_arrivals=scenario.arrivals.model_dump(),
            S=capacity,
            retrieval_setting=scenario.setting,
            cache=scenario.cache,
            retry_policy=scenario.retry_policy,
            faults=profile.faults.model_dump(),
            deadline_ms=scenario.deadline_ms,
            single_worker=True,
        )
        for role in ("standin", "relay"):
            spec = ServiceSpec(
                role=role,
                run_id=run_id,
                ready=directory / f"{role}_ready.json",
                profile=profile.model_dump(),
                policy_path=policy_path,
                events_path=event_path,
                ledger_path=directory / "usage.sqlite3",
                cache_path=directory / "cache.sqlite3",
                base_url=children[0].url + "/openai/v1" if children else None,
            )
            child = OwnedService(spec, directory, settings, clock, on_port)
            children.append(child)
            await child.start()
        standin, relay = children[0].url, children[1].url
        manifest.details["services"] = {
            c.spec.role: {"url": c.url, "owned_pid": c.process.pid} for c in children
        }
        manifest.completion_status = "running"
        persist(manifest, directory, root)
        client = LoadClient(relay, settings, clock)
        async with httpx.AsyncClient(trust_env=False, timeout=settings.admin_timeout_s) as http:
            before, server = await quiescent(http, standin, relay, clock, settings)
            manifest.details.update(initial_standin=before, initial_relay=server)
            base = clock.monotonic_ns()
            base_utc = clock.utc_now()
            manifest.details.update(
                arrival_base_monotonic=base,
                arrival_base_utc=base_utc.isoformat(timespec="milliseconds"),
            )
            admin = asyncio.create_task(
                administer(
                    scenario.admin_events,
                    base,
                    http,
                    standin,
                    relay,
                    clock,
                    settings,
                    admin_writer,
                    run_id,
                )
            )
            report.rows = await dispatch(
                client,
                plans,
                {qid: q.question for qid, q in workload.items()},
                RunContext(
                    run_id=run_id,
                    scenario=scenario.name,
                    source="stand-in",
                    model=profile.model,
                    deadline_ms=scenario.deadline_ms,
                    cache=scenario.cache,
                    fixture_only=scenario.fixture_only,
                ),
                [
                    EventWriter(directory / "client_rows.jsonl", clock=clock),
                    EventWriter(root / "client_rows.jsonl", clock=clock),
                ],
                start_ns=base,
                start_utc=base_utc,
            )
            # Keep the configured run interval, even if the last response finishes early.
            while (remaining := base + int(scenario.duration_s * 1e9) - clock.monotonic_ns()) > 0:
                await asyncio.sleep(remaining / 1e9)
            manifest.details["admin_events"] = await admin
            after, server = await quiescent(http, standin, relay, clock, settings)
            manifest.details.update(
                final_standin=after,
                final_relay=server,
                server_cleanup_observed=True,
                actual_sampling_decisions=after["sampling_audit"],
                logical_requests_recorded=len(report.rows),
                generator_lag=lag_summary(
                    [r.generator_lag_ms for r in report.rows],
                    settings.lag_warning_ms,
                    settings.min_percentile_samples,
                ),
                load_duration_s=(clock.monotonic_ns() - base) / 1e9,
            )
        manifest.completion_status = "complete"
    except PrerequisiteError as exc:
        manifest.completion_status, manifest.reason = "blocked", str(exc)
    except (
        ValueError,
        OSError,
        RuntimeError,
        TimeoutError,
        KeyError,
        TypeError,
        httpx.HTTPError,
    ) as exc:
        manifest.completion_status, manifest.reason = (
            "incomplete",
            f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    except asyncio.CancelledError:
        manifest.completion_status, manifest.reason = "incomplete", "Runner cancelled"
        raise
    finally:
        if admin:
            if not admin.done():
                admin.cancel()
            await asyncio.gather(admin, return_exceptions=True)
        if client:
            await client.close()
        for child in reversed(children):
            await child.stop()
        if event_path.exists():
            selected = [e for e in json_rows([event_path]) if e.get("run") == run_id]
            (directory / "events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in selected), encoding="utf-8"
            )
        if (directory / "admin_events.jsonl").exists():
            manifest.details["admin_event_records"] = json_rows([directory / "admin_events.jsonl"])
        manifest.details["actual_duration_s"] = (clock.monotonic_ns() - started) / 1e9
        manifest.details["owned_children_stopped"] = all(
            c.process is None or c.process.returncode is not None for c in children
        )
        persist(manifest, directory, root)
    return report
