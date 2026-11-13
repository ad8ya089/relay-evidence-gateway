import pytest

from scripts.aggregate import capacity_point, capacity_result, lag_summary, render, scenario_metrics
from tests.fixtures.results import result


def test_linear_percentiles_exclude_failures_and_report_lag() -> None:
    rows = [
        result(
            request_id=f"FAKE-{i}",
            total_ms=t,
            first_token_ms=t / 2,
            scheduled_arrival_monotonic=i * 1_000_000_000,
            completion_monotonic=i * 1_000_000_000 + int(t * 1e6),
        )
        for i, t in enumerate([10, 20, 30, 40])
    ]
    rows.append(
        result(
            request_id="FAKE-failed",
            status="timeout",
            client_status="timeout",
            total_ms=1000,
            first_token_ms=None,
        )
    )
    m = scenario_metrics(rows, {})
    assert m["total_ms"]["p50"] == 25
    assert m["total_ms"]["p95"] == 38.5
    assert m["total_ms"]["p99"] == pytest.approx(39.7)
    assert m["total_ms"]["n"] == 4
    assert m["failed_elapsed_ms"] == [1000]
    assert m["outcomes"]["ok"] == 80
    assert m["outcomes"]["timeout"] == 20
    assert m["token_denominator"] == 4
    assert m["tokens_per_answer"] == 120
    assert lag_summary([0, 10, 30], 20)["rate_not_maintained"]
    assert not lag_summary([0, 10, 20], 20)["rate_not_maintained"]


def test_cost_never_falls_back_to_another_model_and_includes_failed_spend() -> None:
    rows = [result(), result(request_id="FAKE-f", status="error", client_status="error")]
    prices = {
        "prices": {
            "FAKE-model": {
                "date": "2026-01-01",
                "input_per_million": 2,
                "cached_input_per_million": 1,
                "output_per_million": 3,
            }
        }
    }
    assert scenario_metrics(rows, prices)["cost_per_1000"]["value"] == 0.5
    assert (
        scenario_metrics(rows, {"prices": {"other": prices["prices"]["FAKE-model"]}})[
            "cost_per_1000"
        ]["value"]
        is None
    )


def test_capacity_requires_threshold_and_discloses_upper_censoring() -> None:
    rows = [result(request_id=f"FAKE-{i}") for i in range(2)]
    point = capacity_point(rows, 10, 100, 0.99, 2)
    assert point["passes"]
    assert capacity_result([point])["censored"]
    higher = capacity_point([result(status="timeout", client_status="timeout")], 20, 100, 0.99, 2)
    assert not higher["passes"]
    assert capacity_result([point, higher])["value"] == 10
    assert not capacity_result([point, higher])["censored"]


def test_tables_are_deterministic_and_exclude_development_from_official_tables() -> None:
    rows = [result(fixture_only=True)]
    first = render(rows, {}, [], {}, minimum=2, lag_threshold=20)
    assert first == render(rows, {}, [], {}, minimum=2, lag_threshold=20)
    text, diagnostics = first
    assert "Table 1" in text and "Table 2a" in text and "Table 2b" in text
    assert "| steady | not measured" in text
    assert diagnostics
    assert "Fixture-only" in text
