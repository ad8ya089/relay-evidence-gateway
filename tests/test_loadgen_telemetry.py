import json
from pathlib import Path

from relay.app.models import Question
from tests.fixtures.control import controlled


async def test_additive_timing_events_keep_pipeline_and_guard_state_safe(tmp_path: Path) -> None:
    async with controlled(tmp_path) as plane:
        plane.run_id = "FAKE-p6-telemetry"
        outcome = await plane.ask(Question(text="alpha", setting="A", request_id="FAKE-r"))
        assert outcome.status == "ok"
        events = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
        assert all(event["run"] == plane.run_id for event in events)
        spans = [event for event in events if event["ev"] == "stage_timing"]
        assert {"retrieval", "draft", "verification", "deterministic_validation", "rendering"} <= {
            event["stage"] for event in spans
        }
        assert all(event["end_ms"] >= event["start_ms"] for event in spans)
        assert len([e for e in events if e["ev"] == "provider_wait"]) == 2
        state = await plane.state()
        assert state["provider_source"] == "stand-in" and not state["real_guard"]["enabled"]
