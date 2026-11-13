import pytest

from scripts.aggregate import evaluation_metrics
from scripts.build_results_csv import ResultsError, write_csv
from tests.fixtures.results import result


def test_manual_grades_use_actual_and_class_denominators_only_in_tests() -> None:
    rows = [
        result(
            request_id=f"FAKE-{i}",
            record_kind="eval",
            answer_grade=grade,
            answerable=answerable,
            retrieval_setting="B",
        )
        for i, (grade, answerable) in enumerate(
            [
                ("correct", True),
                ("wrong", True),
                ("refused_correctly", False),
                ("refused_wrongly", True),
                ("refused_correctly", False),
            ]
        )
    ]
    e = evaluation_metrics(rows)["B"]
    assert e["asked"] == 5 and e["correct_pct"] == 20 and e["wrong_pct"] == 20
    assert e["unanswerable_n"] == 2 and e["refused_correctly_pct"] == 100
    assert e["performance"]["total_ms"]["n"] == 5
    assert evaluation_metrics([])["A"]["correct_pct"] is None


def test_unknown_answerable_class_and_fixture_grade_persistence(tmp_path: object) -> None:
    from pathlib import Path

    row = result(record_kind="eval", answer_grade="correct", answerable=None)
    assert evaluation_metrics([row])["A"]["refused_correctly_pct"] is None
    with pytest.raises(ResultsError, match="Fixture grades"):
        write_csv([row], Path(str(tmp_path)) / "results.csv")
