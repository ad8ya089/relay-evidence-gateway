"""Synthetic arithmetic/format fixtures only; never official evaluation grades."""

from loadgen.models import ClientRecord
from scripts.result_models import ResultRow


def client(**changes: object) -> ClientRecord:
    data = dict(
        request_id="FAKE-r",
        run_id="FAKE-run",
        scenario="development_smoke",
        source="stand-in",
        question_id="FAKE-q",
        seed=7,
        sampling="uniform",
        sample_draw=0.5,
        fixture_only=True,
        scheduled_arrival_monotonic=0,
        task_dispatch_monotonic=0,
        actual_send_monotonic=1000000,
        first_answer_byte_monotonic=11000000,
        completion_monotonic=21000000,
        scheduled_arrival_utc="2026-11-13T00:00:00.000Z",
        actual_send_utc="2026-11-13T00:00:00.001Z",
        dispatch_timestamp_source="FAKE",
        generator_lag_ms=1,
        transport_wait_ms=1,
        first_token_ms=10,
        total_ms=20,
        status="ok",
        http_status_last=200,
        deadline_ms=8000,
        model="FAKE-model",
        retrieval_setting="A",
        terminal_complete=True,
    )
    return ClientRecord.model_validate(data | changes)


def result(**changes: object) -> ResultRow:
    source = client()
    return ResultRow.model_validate(
        source.model_dump()
        | {
            "start_time": source.actual_send_utc,
            "client_status": "ok",
            "provider_calls": 2,
            "retries": 0,
            "tokens_in": 100,
            "tokens_out": 20,
            "tokens_cached": 10,
            "usage_complete": True,
            "cache_hit": False,
        }
        | changes
    )
