import asyncio
from pathlib import Path

import httpx
import pytest

from loadgen.arrivals import ArrivalConfig, plan
from loadgen.client import LoadClient, dispatch
from loadgen.models import RunContext
from loadgen.settings import load_settings
from loadgen.workload import PrerequisiteError, questions
from relay.obs.clock import SystemClock
from relay.obs.events import EventWriter
from scripts.aggregate import capacity_point, cost
from scripts.build_results_csv import call_metrics
from tests.fixtures.results import result


async def test_client_deadline_cancels_pending_stream_and_keeps_one_record(tmp_path: Path) -> None:
    closed = False

    class Blocked(httpx.AsyncByteStream):
        async def __aiter__(self) -> object:
            await asyncio.Event().wait()
            yield b""

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Blocked())

    clock = SystemClock()
    client = LoadClient(
        "http://127.0.0.1", load_settings(), clock, transport=httpx.MockTransport(handler)
    )
    plans = plan(
        ArrivalConfig(kind="one_shot", count=1),
        ["FAKE"],
        run_id="FAKE",
        seed=1,
        setting="A",
        sampling="uniform",
        maximum=1,
        exponent=1,
    )
    rows = await dispatch(
        client,
        plans,
        {"FAKE": "FAKE question"},
        RunContext(
            run_id="FAKE",
            scenario="fixture-only",
            model="FAKE",
            deadline_ms=25,
            cache=False,
            fixture_only=True,
        ),
        [EventWriter(tmp_path / "rows", clock=clock)],
    )
    await client.close()
    assert len(rows) == 1 and rows[0].status == "timeout"
    assert rows[0].failure_reason == "client_gave_up" and rows[0].total_ms >= 25
    assert rows[0].first_token_ms is None and closed
    assert len((tmp_path / "rows").read_text().splitlines()) == 1


def test_duplicate_overlap_is_recorded_and_sequential_repair_is_not_retry() -> None:
    events = []
    for cid, kind, attempt, start, end in [
        ("1", "draft", 1, 0, 20),
        ("2", "draft", 2, 10, 30),
        ("3", "repair", 1, 30, 40),
    ]:
        events.extend(
            [
                {
                    "ev": "call_start",
                    "call_id": cid,
                    "kind": kind,
                    "attempt": attempt,
                    "t_ms": start,
                },
                {
                    "ev": "call_end",
                    "call_id": cid,
                    "t_ms": end,
                    "total_ms": end - start,
                    "usage": {"in": 10, "out": 2, "cached": 0, "total_tokens": 12},
                },
            ]
        )
    metrics = call_metrics("FAKE", events)
    assert metrics["peak_duplicate_provider_operations"] == 1
    assert metrics["retries"] == 1 and metrics["provider_calls"] == 3


def test_capacity_definition_retains_lag_warning_without_changing_success_rule() -> None:
    rows = [result(request_id=f"FAKE-{i}", generator_lag_ms=30) for i in range(2)]
    point = capacity_point(rows, 60, 100, 0.99, 2)
    assert point["passes"] and point["generator_warning"]


def test_bad_workload_errors_do_not_include_question_contents(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "PRIVATE QUESTION TEXT", "paper": "bad"}')
    with pytest.raises(PrerequisiteError) as error:
        questions(path, load_settings())
    assert "PRIVATE QUESTION TEXT" not in str(error.value)


def test_explicit_free_allocation_and_model_dated_prices_are_required() -> None:
    price = {
        "date": "2026-01-01",
        "input_per_million": 2,
        "cached_input_per_million": 1,
        "output_per_million": 3,
    }
    row = result(source="real", free_allocation=True)
    assert cost([row], {"prices": {"FAKE-model": price}})["billed"] == 0
    assert cost([row], {"prices": {"FAKE-model": price | {"date": "unknown"}}})["value"] is None
