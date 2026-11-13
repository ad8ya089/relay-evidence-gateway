"""Future opt-in burst against an already guarded local Relay; never construct a real provider."""

import json
from pathlib import Path
from uuid import uuid4

import httpx

from loadgen.arrivals import ArrivalConfig, plan
from loadgen.client import LoadClient, dispatch
from loadgen.models import ClientRecord, RunContext
from loadgen.runner import commit_id
from loadgen.settings import load_settings
from loadgen.workload import PrerequisiteError, questions
from relay.app.text import digest
from relay.control.settings import Policy, load_policy
from relay.obs.clock import SystemClock
from relay.obs.events import EventWriter
from relay.providers.groq import is_loopback_url


def preflight(
    *,
    real: bool,
    allow_real: str | None,
    confirmation: str | None,
    n: int,
    state: dict[str, object],
    policy: Policy,
    model: str,
) -> None:
    if not real or allow_real != "1" or confirmation != "RUN":
        raise PrerequisiteError("Real burst requires --real, RELAY_ALLOW_REAL=1 and --confirm RUN")
    if n != 6:
        raise PrerequisiteError("The bounded fidelity burst requires exactly six requests")
    if state.get("provider_source") != "real" or state.get("model") != model:
        raise PrerequisiteError(
            "Existing Relay must explicitly identify the real provider and model"
        )
    if state.get("policy_fingerprint") != digest(policy.model_dump()):
        raise PrerequisiteError("Existing Relay policy differs from the reviewed policy snapshot")
    proof = state.get("real_guard", {})
    if any(proof.get(field) is not True for field in ("enabled", "permitted", "durable")):
        raise PrerequisiteError(
            "Existing Relay must prove enabled, permitted, durable SQLite guardrails"
        )
    attempts = 1 + (
        policy.retry.naive_retries
        if policy.retry.policy == "naive_s7a"
        else policy.retry.max_per_request
    )
    logical_calls = (2 + 2 * policy.normal.repair_max) * attempts
    largest_input = max(max(v.draft_input, v.verify_input) for v in policy.estimates.values())
    tokens_per_call = (
        largest_input
        + max(policy.normal.draft_max_completion_tokens, policy.normal.verify_max_completion_tokens)
        + policy.safety_tokens
    )
    if (
        proof.get("remaining_requests", 0) < n * logical_calls
        or proof.get("remaining_tokens", 0) < n * logical_calls * tokens_per_call
    ):
        raise PrerequisiteError(
            "Remaining real hard caps cannot cover the conservative bounded burst"
        )


async def run(
    *,
    url: str,
    n: int,
    real: bool,
    allow_real: str | None,
    confirmation: str | None,
    model: str,
    root: Path = Path("results"),
    policy_path: Path = Path("config/policy.yaml"),
) -> list[ClientRecord]:
    # Refuse before any HTTP when the explicit intent is absent. No credential lookup.
    if not is_loopback_url(url) or not real or allow_real != "1" or confirmation != "RUN":
        raise PrerequisiteError(
            "Requires explicit local guarded Relay URL, --real, RELAY_ALLOW_REAL=1, --confirm RUN"
        )
    settings, clock, policy = load_settings(), SystemClock(), load_policy(policy_path)
    workload = questions(Path("eval/questions.jsonl"), settings)
    async with httpx.AsyncClient(trust_env=False, timeout=settings.admin_timeout_s) as http:
        response = await http.get(url.rstrip("/") + "/state")
        response.raise_for_status()
        state = response.json()
    preflight(
        real=real,
        allow_real=allow_real,
        confirmation=confirmation,
        n=n,
        state=state,
        policy=policy,
        model=model,
    )
    run_id = f"real-burst-{uuid4().hex}"
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "scenario": "real_burst6",
        "source": "real",
        "model": model,
        "fixture_only": False,
        "git_commit": commit_id(),
        "start_time": clock.utc_now().isoformat(timespec="milliseconds"),
        "profile": "existing guarded real Relay",
        "policy_snapshot": policy.model_dump(),
        "guard_proof": state["real_guard"],
        "services": {"relay": {"url": url, "owned": False}},
        "completion_status": "running",
    }

    def save() -> None:
        for path in (directory / "run_manifest.json", root / "run_manifest.json"):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(path)

    save()
    client = LoadClient(url, settings, clock)
    started = clock.monotonic_ns()
    try:
        plans = plan(
            ArrivalConfig(kind="one_shot", count=n),
            list(workload),
            run_id=run_id,
            seed=policy.retry.seed,
            setting="C",
            sampling="uniform",
            maximum=n,
            exponent=settings.zipf_exponent,
        )
        manifest.update(
            seed=policy.retry.seed,
            deadline_ms=policy.deadline_ms,
            retrieval_setting="C",
            cache=False,
            sampling="uniform",
            question_ids=list(workload),
            planned_requests=[p.model_dump() for p in plans],
            scheduled_duration_s=0,
            offered_arrivals={"kind": "one_shot", "count": n},
            loadgen_snapshot=settings.model_dump(),
            profile_provenance="Existing guarded real Relay; account calibration not inferred",
        )
        save()
        rows = await dispatch(
            client,
            plans,
            {qid: q.question for qid, q in workload.items()},
            RunContext(
                run_id=run_id,
                scenario="real_burst6",
                source="real",
                model=model,
                deadline_ms=policy.deadline_ms,
                cache=False,
                fixture_only=False,
            ),
            [
                EventWriter(root / "client_rows.jsonl", clock=clock),
                EventWriter(directory / "client_rows.jsonl", clock=clock),
            ],
        )
        manifest["completion_status"] = "complete"
        return rows
    finally:
        if manifest["completion_status"] != "complete":
            manifest["completion_status"] = "incomplete"
        await client.close()
        manifest["actual_duration_s"] = (clock.monotonic_ns() - started) / 1e9
        save()
