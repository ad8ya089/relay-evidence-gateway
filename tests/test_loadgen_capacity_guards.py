from pathlib import Path

import pytest

from loadgen.capacity import discover
from loadgen.real_burst import preflight
from loadgen.workload import PrerequisiteError
from relay.app.text import digest
from relay.control.settings import load_policy


@pytest.mark.parametrize("field", ["enabled", "permitted", "durable"])
def test_future_real_burst_requires_every_guard_without_sending_requests(field: str) -> None:
    policy = load_policy()
    proof = {
        "enabled": True,
        "permitted": True,
        "durable": True,
        "remaining_tokens": 10000000,
        "remaining_requests": 10000,
    }
    state = {
        "model": "FAKE",
        "provider_source": "real",
        "policy_fingerprint": digest(policy.model_dump()),
        "real_guard": proof,
    }
    preflight(
        real=True, allow_real="1", confirmation="RUN", n=6, state=state, policy=policy, model="FAKE"
    )
    proof[field] = False
    with pytest.raises(PrerequisiteError, match="guardrails"):
        preflight(
            real=True,
            allow_real="1",
            confirmation="RUN",
            n=6,
            state=state,
            policy=policy,
            model="FAKE",
        )


def test_future_real_caps_and_explicit_intent_are_required() -> None:
    policy = load_policy()
    with pytest.raises(PrerequisiteError, match="requires"):
        preflight(
            real=False,
            allow_real="1",
            confirmation="RUN",
            n=6,
            state={},
            policy=policy,
            model="FAKE",
        )
    state = {
        "model": "FAKE",
        "provider_source": "real",
        "policy_fingerprint": digest(policy.model_dump()),
        "real_guard": {
            "enabled": True,
            "permitted": True,
            "durable": True,
            "remaining_tokens": 1,
            "remaining_requests": 1,
        },
    }
    with pytest.raises(PrerequisiteError, match="hard caps"):
        preflight(
            real=True,
            allow_real="1",
            confirmation="RUN",
            n=6,
            state=state,
            policy=policy,
            model="FAKE",
        )


async def test_capacity_without_human_workload_records_blocked_report(tmp_path: Path) -> None:
    report = await discover(root=tmp_path)
    assert report["completion_status"] == "blocked"
    assert report["kind"] == "S_dev" and report["value"] is None
    assert (tmp_path / "capacity_report.json").exists()
    assert not report["points"]
