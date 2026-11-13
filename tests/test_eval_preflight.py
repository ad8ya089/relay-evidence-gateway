"""Offline dispatch-boundary checks; no provider client is constructed."""

import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval import runner


@pytest.mark.parametrize(
    "change",
    [
        {"observed_at_utc": "2026-11-14T00:00:00"},
        {"observed_at_utc": "invalid"},
        {"remaining_tpd": float("nan")},
        {"remaining_tpd": float("inf")},
        {"remaining_tpd": -1},
        {"remaining_tpm": "unknown"},
        {"model": "other-model"},
    ],
)
def test_invalid_quota_evidence_fails_closed(change: dict[str, object]) -> None:
    evidence = {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "model": "openai/gpt-oss-20b",
        "provenance": "verified account observation",
        "remaining_tpd": 200000,
        "remaining_tpm": 8000,
    }
    plan = runner.real_plan("A", 1, evidence | change)
    assert plan["status"] == "blocked"
    assert any("malformed" in reason for reason in plan["reasons"])


async def test_subset_is_budgeted_and_recorded_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: object) -> None:
        raise AssertionError("Blocked preflight constructed an application")

    monkeypatch.setattr(runner, "create_local_app", unexpected)
    root = await runner.run(
        Namespace(
            real=True,
            settings=["A"],
            quota_evidence=None,
            limit=2,
            output=tmp_path,
            base_url=None,
        )
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "blocked"
    assert manifest["requested"] == 2
    assert len(manifest["question_ids"]) == 2
    assert manifest["plans"][0]["questions"] == 2
    assert manifest["plans"][0]["worst_case_tokens"] == 2 * 14190
    assert manifest["smoke_subset"] is True


async def test_unreadable_quota_file_preserves_blocked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: object) -> None:
        raise AssertionError("Invalid evidence constructed an application")

    monkeypatch.setattr(runner, "create_local_app", unexpected)
    root = await runner.run(
        Namespace(
            real=True,
            settings=["A"],
            quota_evidence=tmp_path / "missing.json",
            limit=None,
            output=tmp_path / "runs",
            base_url=None,
        )
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "blocked"
    assert "unreadable" in manifest["reason"]
