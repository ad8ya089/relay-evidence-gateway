import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from relay.obs.clock import FakeClock
from relay.obs.events import EventContext, EventWriteError, EventWriter


async def test_jsonl_concurrency_and_deterministic_timestamps(tmp_path: Path) -> None:
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))
    context = EventContext(rid="r1", run="test", arrival_ns=clock.monotonic_ns())
    writer = EventWriter(tmp_path / "events.jsonl", clock=clock)
    clock.advance_ms(12.5)
    await asyncio.gather(*[writer.emit(context, "call_start", call_id=f"c{i}") for i in range(50)])
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert len(records) == 50 and len({r["call_id"] for r in records}) == 50
    assert all(r["rid"] == "r1" and r["run"] == "test" and r["t_ms"] == 12.5 for r in records)


def test_nested_redaction_and_body_truncation(tmp_path: Path) -> None:
    key = "FAKE-key-canary-never-a-real-credential"
    writer = EventWriter(tmp_path / "events.jsonl", secrets=[key])
    context = EventContext(rid="r1", run="test", arrival_ns=writer.clock.monotonic_ns())
    writer.write(
        context,
        "call_error",
        nested=[
            {
                "API_KEY": key,
                "headers": {"Authorization": f"Bearer {key}", "safe": "value"},
                "environment": {"any_name": "private-environment-canary"},
                "body_trunc": f"Authorization: Bearer {key}\n" + "x" * 900,
                "message": f"Provider echoed {key}",
            }
        ],
    )
    raw = writer.path.read_text()
    row = json.loads(raw)["nested"][0]
    assert key not in raw and "private-environment-canary" not in raw
    assert "authorization" not in raw.casefold()
    assert len(row["body_trunc"]) <= 500 and row["headers"] == {"safe": "value"}
    assert row["API_KEY"] == "[REDACTED]"


def test_writer_surfaces_persistence_and_serialisation_errors(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path)  # A directory cannot be opened as an event file.
    context = EventContext(rid="r1", run="test", arrival_ns=writer.clock.monotonic_ns())
    with pytest.raises(EventWriteError, match="Cannot persist"):
        writer.write(context, "call_end")
    with pytest.raises(EventWriteError, match="unsupported"):
        writer.write(context, "call_end", unsafe=object())


def test_structured_body_and_alternate_token_fields_are_safe(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    context = EventContext(rid="r", run="test", arrival_ns=writer.clock.monotonic_ns())
    writer.write(
        context,
        "call_error",
        body={"message": "x" * 1000},
        nested={"api-token": "FAKE-token-canary", "bearer_token": "FAKE-bearer-canary"},
    )
    raw = writer.path.read_text()
    assert "FAKE-token-canary" not in raw and "FAKE-bearer-canary" not in raw
    assert len(json.loads(raw)["body"]) == 500
