import csv
from pathlib import Path

import pytest

from eval.grade import AssistedGrade, Grade, save_assisted_grade, save_grade
from eval.runner import real_plan
from scripts.aggregate import quality
from tests.fixtures.results import result


def test_review_provenance_cannot_claim_human_confirmation(tmp_path: Path) -> None:
    values = dict(
        request_id="FAKE",
        question_id="A01",
        retrieval_setting="A",
        answer_grade="wrong",
        failure_stage="model_misread",
    )
    with pytest.raises(ValueError):
        AssistedGrade(**values, human_confirmed=True)
    path = tmp_path / "review.csv"
    save_assisted_grade(AssistedGrade(**values), path)
    raw = next(csv.DictReader(path.open(newline="")))
    assert AssistedGrade.model_validate(raw).human_confirmed is False
    with pytest.raises(ValueError):
        Grade.model_validate(raw)
    with pytest.raises(ValueError):
        save_grade(AssistedGrade(**values), tmp_path / "human.csv")


def test_manual_grade_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "grades.csv"
    save_grade(
        Grade(
            request_id="FAKE",
            question_id="A01",
            retrieval_setting="A",
            answer_grade="correct",
            human_confirmed=True,
        ),
        path,
    )
    assert Grade.model_validate(next(csv.DictReader(path.open(newline="")))).human_confirmed


def test_retry_count_is_shared_and_repair_prompt_grows() -> None:
    plan = real_plan("A", 18)
    assert plan["provider_calls_max_per_question"] == 5
    assert plan["repair_reservation_estimate"] > plan["draft_reservation_estimate"]
    assert plan["status"] == "blocked"


def test_uncertain_review_is_not_a_correct_refusal() -> None:
    row = result(record_kind="eval", answerable=False, review_uncertain=True)
    q = quality([row])
    assert q["graded"] == 0 and q["uncertain"] == 1 and q["refusal_accuracy"] is None


def test_account_screenshot_must_match_reviewed_bytes(tmp_path: Path) -> None:
    import hashlib
    import json

    from scripts.aggregate import verified_account_evidence

    image = tmp_path / "synthetic.png"
    image.write_bytes(b"synthetic test image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "screenshots": [
                    {
                        "filename": image.name,
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    assert verified_account_evidence(manifest) is not None
    image.write_bytes(b"changed synthetic image")
    with pytest.raises(ValueError, match="screenshot"):
        verified_account_evidence(manifest)


async def test_real_evaluation_retains_shared_historical_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace

    from eval import runner

    captured = {}

    def denied_factory(**kwargs: object) -> None:
        captured.update(kwargs)
        raise PermissionError("Synthetic denial; no client is constructed")

    monkeypatch.setattr(runner, "real_plan", lambda *args: {"status": "ready"})
    monkeypatch.setattr(runner, "create_local_app", denied_factory)
    await runner.run(
        Namespace(
            real=True,
            settings=["A"],
            quota_evidence=None,
            limit=None,
            output=tmp_path,
            base_url=None,
        )
    )
    assert captured["ledger_path"] == Path("results/usage.sqlite3")
