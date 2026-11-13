import json
from pathlib import Path

from loadgen.run import invoke, parser


async def test_cli_blocked_official_run_creates_actionable_manifest(tmp_path: Path) -> None:
    args = parser().parse_args(["run", "--scenario", "steady", "--output-root", str(tmp_path)])
    assert await invoke(args) == 2
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["completion_status"] == "blocked"
    assert "human-authored" in manifest["reason"]
    assert not list(tmp_path.rglob("*_ready.json"))


def test_command_parser_exposes_explicit_local_paths() -> None:
    args = parser().parse_args(["capacity", "--fixture-only", "--tiny"])
    assert args.fixture_only and args.tiny
    assert "real-burst" in parser().format_help()
    assert parser().parse_args(["scenarios"]).command == "scenarios"
