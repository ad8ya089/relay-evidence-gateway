import csv
from pathlib import Path

import pytest

from scripts.build_results_csv import ResultsError, build, joined, read_csv, write_csv
from scripts.result_models import COLUMNS, EXTRA, REQUIRED
from tests.fixtures.results import client, result


def calls(unknown: bool = False) -> list[dict[str, object]]:
    events = []
    for index, kind in enumerate(["draft", "verify", "repair", "verify"]):
        events.append(
            {
                "rid": "FAKE-r",
                "ev": "call_start",
                "call_id": str(index),
                "kind": kind,
                "attempt": 1,
                "t_ms": index * 10,
            }
        )
        events.append(
            {
                "rid": "FAKE-r",
                "ev": "call_error" if index == 0 else "call_end",
                "call_id": str(index),
                "total_ms": 5,
                "usage": {}
                if unknown and index == 0
                else {"in": 10, "out": 2, "cached": 1, "total_tokens": 12},
                "t_ms": index * 10 + 5,
            }
        )
    return events


@pytest.mark.parametrize("unknown", [False, True])
def test_all_internal_attempts_become_one_row_with_unknown_usage_preserved(unknown: bool) -> None:
    row = joined(client(), calls(unknown))
    assert row.provider_calls == 4 and row.retries == 0
    assert row.tokens_in == (None if unknown else 40)
    assert row.tokens_out == (None if unknown else 8)
    assert row.known_tokens_in == (30 if unknown else 40)
    assert row.usage_complete != unknown


def test_client_timeout_is_authoritative_over_server_completion() -> None:
    row = joined(
        client(
            status="timeout",
            failure_reason="client_gave_up",
            first_token_ms=None,
            terminal_complete=False,
        ),
        calls() + [{"rid": "FAKE-r", "ev": "done", "status": "ok"}],
    )
    assert row.status == "timeout" and row.first_token_ms is None and row.tokens_in == 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "bad"),
        ("source", "simulated"),
        ("answer_grade", "maybe"),
        ("request_id", ""),
        ("start_time", "yesterday"),
        ("total_ms", -1),
        ("tokens_in", -1),
        ("retries", -1),
        ("retrieval_setting", "D"),
        ("client_status", "timeout"),
    ],
)
def test_validator_refuses_invalid_data(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "results.csv"
    write_csv([result()], path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = str(value)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ResultsError):
        read_csv(path)


def test_required_column_order_duplicates_and_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv([result()], path)
    assert path.read_text().splitlines()[0].split(",")[:31] == REQUIRED + EXTRA
    original = path.read_bytes()
    with pytest.raises(ResultsError):
        write_csv([result(), result()], path)
    assert path.read_bytes() == original
    for columns in (COLUMNS[1:], list(reversed(COLUMNS))):
        path.write_text(",".join(columns) + "\n", encoding="utf-8")
        with pytest.raises(ResultsError):
            read_csv(path)


def test_duplicate_raw_logical_rows_are_not_silently_dropped(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text((client().model_dump_json() + "\n") * 2, encoding="utf-8")
    with pytest.raises(ResultsError, match="FAKE-r"):
        build([path], [], [])
