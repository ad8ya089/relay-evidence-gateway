import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from relay.control.ledger import Ledger
from relay.obs.clock import FakeClock
from relay.obs.events import EventWriter
from scripts import limit_probe, probe_groq
from scripts.probe_common import ProbeError, ProbeSession, probe_settings
from tests.fake_groq_server import FakeGroqServer
from tests.test_provider_http import request


def arguments(
    module: ModuleType, server: FakeGroqServer, tmp_path: Path, *extra: str
) -> argparse.Namespace:
    return module.parser().parse_args(
        [
            "--local",
            "--base-url",
            server.base_url,
            "--confirm",
            "RUN",
            "--ledger",
            str(tmp_path / "usage.sqlite3"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--output",
            str(tmp_path / ("shapes.json" if module is probe_groq else "limits.jsonl")),
            *extra,
        ]
    )


async def test_shape_probe_against_fake_http_including_opt_in_experiment(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    args = arguments(probe_groq, fake_server, tmp_path, "--experiment-stream-schema")
    assert await probe_groq.run(args) == 0
    artifact = json.loads(args.output.read_text())
    assert len(artifact["observed"]) == fake_server.request_count == 3
    assert all(row["source"] == "stand-in" for row in artifact["observed"])
    assert all(row["outcome"] == "ok" for row in artifact["observed"])
    assert artifact["observed"][0]["provider_request_id"] == "FAKE-provider-request"
    assert "usage" in artifact["observed"][0]["response_fields"]
    assert artifact["unverified"] and artifact["documented"]["structured_outputs_streaming"]
    assert fake_server.payloads[1]["response_format"] == {"type": "json_object"}
    assert fake_server.payloads[2]["response_format"]["type"] == "json_schema"
    assert all(payload["reasoning_effort"] == "low" for payload in fake_server.payloads)
    assert "authorization" not in args.output.read_text().casefold()


@pytest.mark.parametrize("script", ["scripts.probe_groq", "scripts.limit_probe"])
async def test_dry_run_entrypoints_never_execute_even_with_real_flag(
    fake_server: FakeGroqServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
) -> None:
    monkeypatch.setenv("RELAY_ALLOW_REAL", "1")
    key = "FAKE-credential-in-environment-never-used"
    monkeypatch.setenv("GROQ_API_KEY", key)
    output = tmp_path / "never-created.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        script,
        "--real",
        "--dry-run",
        "--base-url",
        fake_server.base_url,
        "--output",
        str(output),
        "--ledger",
        str(tmp_path / "never-created.sqlite3"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == 0, stderr.decode()
    assert b"Dry run" in stdout and key.encode() not in stdout + stderr
    assert not output.exists() and not (tmp_path / "never-created.sqlite3").exists()
    assert fake_server.request_count == 0


def test_default_plan_is_bounded_and_account_limits_are_unknown() -> None:
    args = limit_probe.parser().parse_args([])
    config = limit_probe.configured(args, probe_settings())
    plan, tiny, large = limit_probe.build_plan(args, config)
    assert plan.mode == "dry-run" and plan.max_requests == 56
    assert all(value is None for value in plan.account_limits.model_dump().values())
    assert config.tpm_input_tokens == 3000
    assert plan.phases[1].estimated_tokens <= 10000 and plan.phases[1].concurrency == 3
    assert tiny.token_reservation and large.token_reservation
    with pytest.raises(ProbeError, match="10K"):
        limit_probe.build_plan(args, config.model_copy(update={"tpm_input_tokens": 10000}))


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--rpm-requests", "41"),
        ("--concurrency-requests", "9"),
        ("--tpm-requests", "4"),
        ("--rpm-rates", "2,1"),
    ],
)
def test_invalid_or_excessive_probe_configuration_is_rejected(flag: str, value: str) -> None:
    args = limit_probe.parser().parse_args([flag, value])
    with pytest.raises((ValueError, ProbeError)):
        limit_probe.configured(args, probe_settings())


async def test_limit_probe_respects_429_and_does_not_call_rpm_again(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    fake_server.modes = ["unknown", "success"]
    fake_server.retry_after = "0.01"
    args = arguments(
        limit_probe,
        fake_server,
        tmp_path,
        "--rpm-requests",
        "2",
        "--rpm-rates",
        "60000",
        "--tpm-requests",
        "2",
        "--tpm-input-tokens",
        "20",
        "--concurrency-requests",
        "2",
        "--recovery-requests",
        "2",
        "--summary",
        str(tmp_path / "summary.json"),
    )
    assert await limit_probe.run(args) == 0
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert fake_server.request_count == len(rows) == 5
    assert sum(row["phase"] == "rpm" for row in rows) == 1
    assert rows[0]["retry_after_ms"] == 10 and rows[0]["error_kind"] == "rate_limit_unknown"
    assert all(row["source"] == "stand-in" and row["observed_against"] != "Groq" for row in rows)
    summary = json.loads(args.summary.read_text())
    assert summary["observed_recovery_ms"] >= 10
    with Ledger(args.ledger) as ledger:
        assert ledger.session_usage(summary["run_id"], real=False).attempts == 5
        assert ledger.requests_used_today(probe_settings().model) == 0


async def test_recovery_is_bounded_and_long_retry_after_aborts(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    fake_server.mode, fake_server.retry_after = "unknown", "0"
    args = arguments(
        limit_probe,
        fake_server,
        tmp_path,
        "--rpm-requests",
        "2",
        "--rpm-rates",
        "60000",
        "--tpm-requests",
        "2",
        "--tpm-input-tokens",
        "20",
        "--concurrency-requests",
        "2",
        "--recovery-requests",
        "2",
        "--summary",
        str(tmp_path / "summary.json"),
    )
    assert await limit_probe.run(args) == 0
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert sum(row["phase"] == "recovery" for row in rows) == 2
    assert fake_server.request_count == 7
    fake_server.retry_after = "1000"
    with pytest.raises(ProbeError, match="bounded wait"):
        await limit_probe.run(args)
    assert fake_server.request_count == 8  # Only the initial 429, never an early follow-up.


async def test_cooldown_honours_all_out_of_order_observations(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    args = arguments(limit_probe, fake_server, tmp_path)
    config = probe_settings()
    plan, _, _ = limit_probe.build_plan(args, config)
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))
    session = ProbeSession(args, plan, clock=clock)
    session.last_429_ns = 3_000_000_000
    session.rate_observations = [(0, 10000), (3_000_000_000, 1000)]
    clock.advance_ms(3000)
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)
        clock.advance_ms(seconds * 1000)

    await session.cooldown(config, sleep)
    assert waited == [7.0]
    session.daily_limit_seen = True
    with pytest.raises(ProbeError, match="Daily quota"):
        await session.cooldown(config, sleep)


def test_real_limit_probe_refuses_missing_account_evidence(tmp_path: Path) -> None:
    args = limit_probe.parser().parse_args(["--real"])
    config = probe_settings()
    plan, _, _ = limit_probe.build_plan(args, config)
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        with pytest.raises(ProbeError, match="evidence"):
            limit_probe.check_daily_headroom(config, plan, ledger)


async def test_probe_cancellation_is_saved_before_it_propagates(
    fake_server: FakeGroqServer,
    tmp_path: Path,
) -> None:
    fake_server.mode = "hang"
    args = arguments(limit_probe, fake_server, tmp_path)
    plan, _, _ = limit_probe.build_plan(args, probe_settings())
    async with ProbeSession(args, plan) as session:
        session.output_events = EventWriter(
            args.output, clock=session.clock, sanitizer=session.provider.sanitizer
        )
        pending = asyncio.create_task(session.call("cancel-fixture", request(stream=True)))
        await asyncio.wait_for(fake_server.entered.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(fake_server.disconnected.wait(), 2)
    row = json.loads(args.output.read_text().splitlines()[0])
    assert row["error_kind"] == "cancelled" and row["outcome"] == "error"


async def test_local_mode_cannot_target_an_external_provider(tmp_path: Path) -> None:
    args = probe_groq.parser().parse_args(
        [
            "--local",
            "--base-url",
            "https://api.groq.com/openai/v1",
            "--confirm",
            "RUN",
            "--ledger",
            str(tmp_path / "never.sqlite3"),
        ]
    )
    with pytest.raises(ProbeError, match="Local mode"):
        await probe_groq.run(args)
    assert not (tmp_path / "never.sqlite3").exists()


@pytest.mark.parametrize("script", ["scripts.probe_groq", "scripts.limit_probe"])
async def test_local_execution_entrypoints_use_only_fake_http(
    fake_server: FakeGroqServer,
    tmp_path: Path,
    script: str,
) -> None:
    output = tmp_path / ("shapes.json" if script.endswith("probe_groq") else "limits.jsonl")
    args = [
        sys.executable,
        "-m",
        script,
        "--local",
        "--base-url",
        fake_server.base_url,
        "--confirm",
        "RUN",
        "--output",
        str(output),
        "--ledger",
        str(tmp_path / "usage.sqlite3"),
        "--events",
        str(tmp_path / "events.jsonl"),
    ]
    if script.endswith("limit_probe"):
        args += [
            "--rpm-requests",
            "2",
            "--rpm-rates",
            "60000",
            "--tpm-requests",
            "2",
            "--tpm-input-tokens",
            "20",
            "--concurrency-requests",
            "2",
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == 0, (stdout + stderr).decode()
    assert output.exists() and fake_server.request_count in {2, 6}
    assert "authorization" not in output.read_text().casefold()
