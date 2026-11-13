"""Assemble local evidence without fabricating missing screenshots or conversation exports."""

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from eval.grade import Grade


def build(root: Path = Path("relay-submission")) -> None:
    if root.exists() and not (root / ".relay-package.json").exists():
        raise ValueError("Refusing to overwrite a directory not owned by this package builder")
    root.mkdir(exist_ok=True)
    (root / ".relay-package.json").write_text(json.dumps({"repository": "relay-evidence-gateway"}))
    for name in ("code", "raw", "conversation_history"):
        (root / name).mkdir(exist_ok=True)
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().strip("\0").split("\0")
    for name in tracked:
        if (
            name.startswith("results/")
            or name == ".env"
            or name.endswith((".sqlite3", ".onnx", ".pyc"))
        ):
            continue
        target = root / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    p7 = Path("results/p7")
    for name in ("writeup.pdf", "writeup.md", "results.csv"):
        shutil.copyfile(p7 / name, root / name)
    raw = root / "raw"
    for src, name in [
        (p7 / "tables.json", "tables.json"),
        (Path("results/connectivity/first_real_result.json"), "connectivity_result.json"),
        (p7 / "metrics.json", "metrics.json"),
        (p7 / "calibration.json", "calibration.json"),
        (Path("data/corpus_manifest.json"), "corpus_manifest.json"),
        (Path("results/client_rows.jsonl"), "client_rows.jsonl"),
    ]:
        shutil.copyfile(src, raw / name)
    events = [
        Path("results/events.jsonl"),
        p7 / "real_smoke/events.jsonl",
        *sorted(p7.glob("standin_evaluation*/eval-*/events.jsonl")),
    ]
    with (raw / "events.jsonl").open("wb") as output:
        for path in events:
            output.write(path.read_bytes())
    probe = Path("results/limit_probe.jsonl")
    (raw / "limit_probe.jsonl").write_bytes(probe.read_bytes() if probe.exists() else b"")
    grades = p7 / "grades.csv"
    if grades.exists():
        shutil.copyfile(grades, raw / "grades.csv")
    else:
        with (raw / "grades.csv").open("w", newline="", encoding="utf-8") as output:
            csv.DictWriter(output, fieldnames=list(Grade.model_fields)).writeheader()
    manifests = sorted(p7.glob("**/manifest.json"))
    (raw / "run_manifest.json").write_text(
        json.dumps(
            {
                "runs": [json.loads(p.read_text()) for p in manifests],
                "real_quota_probes": "not measured; limit_probe.jsonl deliberately empty",
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
    )
    for directory in ["real_smoke", "standin_evaluation", "standin_evaluation_repeat"]:
        for path in (p7 / directory).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml"}:
                target = raw / directory / path.relative_to(p7 / directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    (root / "README.md").write_text(
        "# Relay submission - evidence incomplete\n\n"
        "Run code with uv sync --locked --extra dev; make test.\n"
        "Real execution requires explicit provider selection, permission and --real.\n"
        "Tables come only from scripts.aggregate, CSV and captured provider evidence.\n"
        "The real smoke safely abstained; it is not a successful evaluated answer.\n"
        "Stand-in evaluations are ungraded; historical failures are retained.\n"
        "Missing: human grades, real Limits screenshot, conversation exports and Vaani feedback.\n"
        "Official calibrated scenarios and real A/B/C quality are not measured.\n"
        "An empty raw/limit_probe.jsonl means no quota probe was performed.\n"
        "Run python -m scripts.check_submission from the repository root.\n"
    )
    (root / "conversation_history/README.md").write_text(
        "Candidate action required: add full actual uncleaned exports; "
        "no transcript was invented.\n"
    )
    files = [
        {
            "path": str(p.relative_to(root)).replace("\\", "/"),
            "size": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "provenance": "tracked repository source"
            if "code" in p.relative_to(root).parts
            else "recorded/generated local evidence; missing items disclosed",
        }
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    ]
    (root / "artifact_manifest.json").write_text(json.dumps({"files": files}, indent=2))
    print(f"Packaged {len(files)} files; candidate inputs remain explicitly missing")


if __name__ == "__main__":
    build()
