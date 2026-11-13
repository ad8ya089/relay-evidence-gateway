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


def test_header_only_grades_file_does_not_gain_a_second_header(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "grades.csv"
    with path.open("w", newline="") as output:
        csv.DictWriter(output, fieldnames=list(Grade.model_fields)).writeheader()
    grade = Grade(
        request_id="FAKE",
        question_id="A01",
        retrieval_setting="A",
        answer_grade="correct",
        human_confirmed=True,
    )
    save_grade(grade, path)
    with path.open(newline="") as source:
        assert len(list(csv.DictReader(source))) == 1
    with pytest.raises(ValueError, match="already"):
        save_grade(
            grade.model_copy(update={"failure_stage": "model_misread"}), tmp_path / "grades.csv"
        )
