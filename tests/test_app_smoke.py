import asyncio
import json
import sys
from pathlib import Path

import pytest


async def invoke(tmp_path: Path, *extra: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "relay.app.smoke",
        "--question",
        "Explain encoder-decoder attention",
        "--setting",
        "A",
        "--ledger",
        str(tmp_path / "usage.sqlite3"),
        "--events",
        str(tmp_path / "events.jsonl"),
        *extra,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    assert process.returncode is not None
    return process.returncode, stdout.decode(), stderr.decode()


async def test_local_cli_uses_actual_corpus_and_p3_without_credentials(tmp_path: Path) -> None:
    if not Path("data/chunks/A.jsonl").exists():
        pytest.skip("Pinned corpus unavailable; smoke requires P1 ingestion")
    code, stdout, stderr = await invoke(tmp_path, "--deadline-ms", "30000")
    assert code == 0, stdout + stderr
    result = json.loads(stdout)
    assert result["source"] == "stand-in" and result["state"] == "RELEASED"
    assert result["provider_calls"] == 2 and result["usage"]["complete"]
    assert result["citations"][0]["arxiv_version"] == 7
    assert result["first_token_ms"] >= result["provider_ttft_ms"]


async def test_cli_rejects_external_provider_before_opening_ledger(tmp_path: Path) -> None:
    code, _, _ = await invoke(tmp_path, "--base-url", "https://api.groq.com/openai/v1")
    assert code == 2 and not (tmp_path / "usage.sqlite3").exists()
