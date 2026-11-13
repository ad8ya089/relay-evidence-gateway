import pytest

from scripts.timeline import decompose
from tests.fixtures.results import client


def test_actual_retry_is_positive_and_repair_is_work_without_overlap() -> None:
    row = client(total_ms=100, first_token_ms=80)
    events = [
        {"ev": "arrived", "server_arrival_monotonic": row.actual_send_monotonic},
        {"ev": "execution_start", "t_ms": 10, "queue_wait_ms": 10},
        {"ev": "stage_timing", "stage": "retrieval", "start_ms": 10, "end_ms": 20},
        {"ev": "provider_wait", "start_ms": 20, "end_ms": 25},
        {"ev": "call_start", "call_id": "1", "t_ms": 25, "kind": "draft", "attempt": 1},
        {"ev": "call_error", "call_id": "1", "t_ms": 35},
        {"ev": "retry_wait", "start_ms": 35, "end_ms": 45},
        {"ev": "call_start", "call_id": "2", "t_ms": 45, "kind": "draft", "attempt": 2},
        {"ev": "call_end", "call_id": "2", "t_ms": 65},
        {"ev": "call_start", "call_id": "3", "t_ms": 65, "kind": "repair", "attempt": 1},
        {"ev": "call_end", "call_id": "3", "t_ms": 75},
        {"ev": "stage_timing", "stage": "deterministic_validation", "start_ms": 75, "end_ms": 80},
        {"ev": "first_delta", "what": "content", "provider_elapsed_ms": 3},
    ]
    timeline = decompose(row, events)
    assert timeline["waiting_ms"] == 15
    assert timeline["working_ms"] == 35
    assert timeline["retrying_ms"] == 30
    assert timeline["unknown_ms"] == 20
    assert sum(timeline[k + "_ms"] for k in ("waiting", "working", "retrying", "unknown")) == 100
    assert timeline["provider_ttft_ms"] == 3
    assert timeline["user_visible_first_token_ms"] == 80
    assert timeline["after_first_token_ms"] == 20
    assert timeline["principal_delay"] == "before"


def test_incomplete_timeline_stays_unknown_and_invalid_interval_rejected() -> None:
    assert decompose(client(), [])["unknown_ms"] == 20
    assert decompose(client(), [])["missing_events"]
    with pytest.raises(ValueError, match="Invalid event interval"):
        decompose(client(), [{"ev": "retry_wait", "start_ms": 10, "end_ms": 0}])
