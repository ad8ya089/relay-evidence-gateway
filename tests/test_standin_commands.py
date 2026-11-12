import asyncio
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("name,code", [("development", 0), ("free_exact", 2), ("scaled", 2)])
async def test_startup_check_is_local_and_exact_profiles_refuse_missing_values(
    name: str, code: int
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "standin.service",
        "--profile",
        name,
        "--check",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == code, stderr.decode()
    if code:
        assert b"rpm, tpm, rpd, tpd" in stdout
    else:
        assert b"startup valid" in stdout
    assert b"Authorization" not in stdout + stderr


async def test_calibration_cli_without_real_events_is_uncalibrated(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "standin.calibrate",
        "--events",
        str(tmp_path / "missing.jsonl"),
        "--output",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode == 0, stderr.decode()
    report = json.loads(output.read_text())
    assert report["status"] == "uncalibrated" and report["n_calls"] == {"draft": 0, "verify": 0}
    assert b"p99 is unreliable" in stdout
    assert report["alpha"]["source"] == "assumed"
    assert report["faults"]["server_error_prob"]["value"] == 0.025
