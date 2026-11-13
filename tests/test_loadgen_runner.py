import json
from pathlib import Path

import httpx

from loadgen.runner import administer, execute
from loadgen.scenarios import AdminEvent, catalog
from loadgen.settings import load_settings
from relay.obs.clock import SystemClock
from relay.obs.events import EventWriter
from scripts.build_results_csv import build, json_rows, read_csv, write_csv


async def test_owned_development_http_smoke_records_admitted_and_rejected(
    tmp_path: Path, loopback_ports: set[int]
) -> None:
    report = await execute(
        catalog()[1]["development_smoke"], root=tmp_path, on_port=loopback_ports.add
    )
    manifest = json.loads(report.manifest_path.read_text())
    assert manifest["completion_status"] == "complete", manifest.get("reason")
    assert manifest["owned_children_stopped"]
    assert manifest["server_cleanup_observed"]
    assert len(report.rows) == 8
    assert any(r.status == "rate_limited" for r in report.rows)
    assert any(r.status == "ok" for r in report.rows)
    assert len(json_rows([tmp_path / "client_rows.jsonl"])) == 8
    rows = build(
        [tmp_path / "client_rows.jsonl"], [tmp_path / "events.jsonl"], [report.manifest_path]
    )
    write_csv(rows, tmp_path / "results.csv")
    assert len(read_csv(tmp_path / "results.csv")) == 8
    assert all(not r.answer_grade and r.fixture_only for r in rows)
    assert all(s["url"].startswith("http://127.0.0.1:") for s in manifest["services"].values())


async def test_blocked_run_has_manifest_without_services(tmp_path: Path) -> None:
    report = await execute(
        catalog()[1]["steady"], root=tmp_path, capacity_path=tmp_path / "missing"
    )
    assert report.manifest.completion_status == "blocked"
    assert report.manifest_path.exists()
    assert not list(tmp_path.rglob("*_ready.json"))


async def test_admin_failure_is_logged_and_stops_later_actions(tmp_path: Path) -> None:
    clock = SystemClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"model": "FAKE", "models": {"FAKE": {"limits": {"rpm": 100, "tpm": 200}}}},
            )
        return httpx.Response(500)

    import pytest

    path = tmp_path / "admin.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RuntimeError, match="admin event failed"):
            await administer(
                [AdminEvent(offset_s=0, action="halve_minute_limits", factor=0.5)],
                clock.monotonic_ns(),
                http,
                "http://127.0.0.1",
                "http://127.0.0.1",
                clock,
                load_settings(),
                EventWriter(path, clock=clock),
                "FAKE",
            )
    records = json_rows([path])
    assert records[0]["response_status"] == 500
    assert records[0]["status"] == "incomplete"
