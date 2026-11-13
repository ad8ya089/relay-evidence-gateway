import json
from pathlib import Path

import pytest

from eval.dataset import QUESTIONS, load_questions, validate_dataset


def test_candidate_distribution_and_model_facing_isolation() -> None:
    assert validate_dataset() == {
        "questions": 18,
        "answerable": 12,
        "unanswerable": 6,
        "grades": "blank",
    }
    rows = load_questions()
    assert [q.qid for q in rows] == [q.qid for q in load_questions()]
    assert all(
        set(q.model_dump()) == {"qid", "question", "paper", "arxiv_id", "arxiv_version"}
        for q in rows
    )


def test_duplicate_and_wrong_version_refused(tmp_path: Path) -> None:
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    for change in ("duplicate", "version"):
        data = [r.copy() for r in rows]
        if change == "duplicate":
            data[0] = data[1]
        else:
            data[0]["arxiv_version"] = 99
        path = tmp_path / "bad.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in data))
        with pytest.raises(ValueError):
            load_questions(path)
