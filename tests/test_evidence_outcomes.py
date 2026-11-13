from pathlib import Path

import pytest

from scripts.aggregate import submission_tables
from scripts.build_results_csv import read_csv, write_csv
from scripts.result_models import REQUIRED
from tests.fixtures.results import result


@pytest.mark.parametrize(
    "statuses", [[], ["error"], ["ok", "error"], ["rate_limited", "timeout", "error"]]
)
def test_submission_outcome_denominator(statuses: list[str]) -> None:
    rows = [
        result(
            request_id=f"FAKE-{i}",
            scenario="real_smoke",
            fixture_only=False,
            status=status,
            client_status=status,
        )
        for i, status in enumerate(statuses)
    ]
    tables, _ = submission_tables(rows)
    table = tables["Table 2b: Outcomes"]
    row = next(r for r in table["rows"] if r[0] == "real_smoke")
    assert table["headers"][4] == "Error %"
    if statuses:
        assert sum(float(v) for v in row[1:5]) == pytest.approx(100, abs=0.2)
        assert float(row[4]) == pytest.approx(
            100 * statuses.count("error") / len(statuses), abs=0.1
        )
    else:
        assert row[1:5] == ["not measured"] * 4


def test_console_provenance_does_not_claim_screenshot() -> None:
    tables, _ = submission_tables([])
    rows = tables["Table 1: Limits found in the real service"]["rows"]
    assert rows[0][1] == "30" and rows[3][1] == "200000"
    assert all("Candidate-reported" in rows[i][2] and "missing" in rows[i][3] for i in [0, 3])


def test_csv_prefix_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv([result()], path)
    assert path.read_text().splitlines()[0].split(",")[: len(REQUIRED)] == REQUIRED
    assert len(read_csv(path)) == 1
