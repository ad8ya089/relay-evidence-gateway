from pathlib import Path

import pytest

from eval.grade import Grade, save_grade


def test_grade_requires_human_selection_and_failure_context(tmp_path: Path) -> None:
    values = dict(request_id="FAKE", question_id="A01", retrieval_setting="A")
    with pytest.raises(ValueError):
        Grade(**values)
    with pytest.raises(ValueError):
        Grade(**values, answer_grade="correct", human_confirmed=False)
    grade = Grade(**values, answer_grade="wrong", human_confirmed=True)
    with pytest.raises(ValueError, match="stage"):
        save_grade(grade, tmp_path / "grades.csv")
    assert not (tmp_path / "grades.csv").exists()
    save_grade(grade.model_copy(update={"failure_stage": "model_misread"}), tmp_path / "grades.csv")
    with pytest.raises(ValueError, match="already"):
        save_grade(
            grade.model_copy(update={"failure_stage": "model_misread"}), tmp_path / "grades.csv"
        )
