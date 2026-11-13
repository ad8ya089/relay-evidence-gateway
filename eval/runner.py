"""Sequential normal-pipeline evaluation; runtime imports questions, never the answer key."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from eval.dataset import load_questions, validate_dataset
from relay.app.api import create_local_app
from relay.app.delivery import iter_validated_answer
from relay.app.models import Question
from relay.control.settings import load_policy
from relay.obs.clock import SystemClock
from relay.obs.events import EventContext, EventWriter
from relay.settings import SafetySettings


def real_plan(
    setting: str, count: int, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    policy, safety = load_policy(), SafetySettings()
    e = policy.estimates[setting]
    draft = e.draft_input + policy.normal.draft_max_completion_tokens + policy.safety_tokens
    verifier = e.verify_input + policy.normal.verify_max_completion_tokens + policy.safety_tokens
    # The repair carries the rejected completion and complaint/envelope overhead.
    # This is a conservative estimate; actual serialized messages are reserved again at dispatch.
    repair_growth = policy.normal.draft_max_completion_tokens + policy.safety_tokens
    repair = draft + repair_growth
    initial = draft + verifier
    calls = 2 * (1 + policy.normal.repair_max) + policy.retry.max_per_request
    per_question = initial + policy.normal.repair_max * (repair + verifier)
    per_question += policy.retry.max_per_request * max(draft, repair, verifier)
    worst = count * per_question
    approval = Path("eval/datasets/candidate-final/provenance.json")
    reasons = []
    if (
        not approval.exists()
        or json.loads(approval.read_text()).get("candidate_approval") != "approved"
    ):
        reasons.append("Final candidate dataset approval is not recorded")
    if worst > safety.real_token_cap:
        reasons.append(
            "Worst-case draft/verifier/repair/retry estimate exceeds existing session token cap"
        )
    if count * calls > safety.real_request_cap:
        reasons.append("Worst-case provider attempts exceed existing request cap")
    if not evidence:
        reasons.append("Fresh verified organization remaining quota is not supplied")
    else:
        observed = datetime.fromisoformat(str(evidence["observed_at_utc"]))
        age = (datetime.now(UTC) - observed).total_seconds()
        if observed.utcoffset() is None or not 0 <= age <= policy.window_s:
            reasons.append("Quota evidence is stale or future-dated")
        if (
            evidence.get("model") != "openai/gpt-oss-20b"
            or evidence.get("provenance") != "verified account observation"
        ):
            reasons.append("Quota evidence lacks matching verified model provenance")
        if worst > float(evidence.get("remaining_tpd", 0)) * 0.7:
            reasons.append("Run exceeds 70% of verified remaining daily tokens")
        if initial > float(evidence.get("remaining_tpm", 0)):
            reasons.append("Initial pair does not fit observed minute quota")
    return {
        "setting": setting,
        "questions": count,
        "initial_estimate": initial,
        "draft_reservation_estimate": draft,
        "verifier_reservation_estimate": verifier,
        "repair_reservation_estimate": repair,
        "repair_prompt_growth_allowance": repair_growth,
        "shared_retry_max": policy.retry.max_per_request,
        "provider_calls_max_per_question": calls,
        "per_question_estimate_with_repair_retry": per_question,
        "reasoning_allowance": "Included in completion caps; not additional output tokens",
        "enforced_session_reservation_cap": safety.real_token_cap,
        "estimate_status": "o200k approximation; not a provider-guaranteed input bound",
        "worst_case_tokens": worst,
        "session_cap": safety.real_token_cap,
        "status": "blocked" if reasons else "ready",
        "reasons": reasons,
    }


async def run(args: argparse.Namespace) -> Path:
    questions = load_questions()  # This file contains no gold facts or answerability.
    run_id = "eval-" + uuid4().hex
    root = args.output / run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "source": "real" if args.real else "stand-in",
        "evaluation": True,
        "settings": args.settings,
        "question_ids": [q.qid for q in questions],
        "requested": len(questions),
        "smoke_subset": args.limit is not None,
        "status": "preparing",
        "grades": "blank",
        "plans": [],
    }
    if args.real:
        evidence = json.loads(args.quota_evidence.read_text()) if args.quota_evidence else None
        manifest["plans"] = [real_plan(s, len(questions), evidence) for s in args.settings]
        if len(args.settings) != 1 or any(p["status"] != "ready" for p in manifest["plans"]):
            manifest["status"] = "blocked"
            manifest["reason"] = (
                "Real settings require separate sequential invocations and fresh quota checks"
            )
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            return root
    if not args.real and not args.base_url:
        raise ValueError("Stand-in evaluation needs an explicit existing loopback --base-url")
    if not args.real and not args.base_url.rstrip("/").endswith("/openai/v1"):
        manifest.update(status="blocked", reason="Provider base URL must include /openai/v1")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        raise ValueError("Provider base URL must include /openai/v1")
    clock = SystemClock()
    writer = EventWriter(root / "rows.jsonl", clock=clock)
    policy_path = Path("config/policy.yaml")
    if args.real:
        config = yaml.safe_load(Path("config/p7.yaml").read_text())
        data = load_policy().model_dump()
        data.update(
            provider="groq", daily_reset_known=False, max_inflight_real=1, max_inflight_provider=1
        )
        data["limits"] = {config["model"]: config["limits"]}
        policy_path = root / "policy.yaml"
        policy_path.write_text(yaml.safe_dump(data))
    try:
        app = create_local_app(
            policy_path=policy_path,
            real=args.real,
            base_url=args.base_url,
            no_cache=True,
            ledger_path=Path("results/usage.sqlite3") if args.real else root / "usage.sqlite3",
            cache_path=root / "cache.sqlite3",
            events_path=root / "events.jsonl",
            run_id=run_id,
        )
    except (ValueError, PermissionError, OSError) as exc:
        manifest.update(status="blocked", reason="Application preflight: " + type(exc).__name__)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return root
    try:
        async with app.router.lifespan_context(app):
            plane = app.state.control
            for setting in args.settings:
                for q in questions[: args.limit]:
                    started = clock.monotonic_ns()
                    stamp = clock.utc_now().isoformat(timespec="milliseconds")
                    outcome = await plane.ask(
                        Question(
                            text=q.question,
                            setting=setting,
                            request_id=f"{run_id}-{setting}-{q.qid}",
                        ),
                        cache_enabled=False,
                    )
                    if hasattr(outcome, "releasable"):
                        visible = (
                            "".join(
                                [
                                    p
                                    async for p in iter_validated_answer(
                                        outcome, clock=clock, events=plane.events
                                    )
                                ]
                            )
                            if outcome.releasable
                            else ""
                        )
                        row = outcome.model_dump(mode="json") | {"answer": visible}
                    else:
                        row = {
                            "status": "rate_limited",
                            "failure_reason": outcome.reason,
                            "answer": "",
                            "provider_calls": 0,
                            "citations": [],
                        }
                    row.update(
                        question_id=q.qid,
                        question=q.question,
                        paper=q.paper,
                        source="real" if args.real else "stand-in",
                        retrieval_setting=setting,
                        run_id=run_id,
                        request_id=f"{run_id}-{setting}-{q.qid}",
                        start_time=stamp,
                        total_ms=(clock.monotonic_ns() - started) / 1e6,
                        answer_grade="",
                        provider_calls=getattr(outcome, "provider_calls", 0),
                    )
                    writer.write(
                        EventContext(rid=row["request_id"], run=run_id, arrival_ns=started),
                        "evaluation_result",
                        **row,
                    )
        manifest["status"] = "complete_subset" if args.limit else "complete"
    except (ValueError, RuntimeError, OSError) as exc:
        manifest.update(status="incomplete", reason=type(exc).__name__)
    finally:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--settings", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--base-url")
    parser.add_argument("--quota-evidence", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/p7/eval"))
    args = parser.parse_args()
    validate_dataset()
    if args.limit is not None and not 1 <= args.limit <= 18:
        parser.error("limit must be 1..18; subsets are never official complete evaluations")
    root = asyncio.run(run(args))
    print(root)
    return 2 if json.loads((root / "manifest.json").read_text())["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
