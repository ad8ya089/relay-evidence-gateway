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
    shutil.copyfile(Path("results/results.csv"), raw / "p6_results.csv")
    shutil.copyfile(p7 / "historical_hashes.json", raw / "historical_hashes.json")
    shutil.copyfile(Path("results/capacity_report.json"), raw / "p6_capacity_report.json")
    for path in Path("results/runs").glob("*/run_manifest.json"):
        target = raw / "p6_runs" / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    events = [
        Path("results/events.jsonl"),
        Path("results/evidence_completion/experiments/events.jsonl"),
        Path("results/final_completion/experiments/events.jsonl"),
        p7 / "real_smoke/events.jsonl",
        *sorted(p7.glob("standin_evaluation*/eval-*/events.jsonl")),
    ]
    with (raw / "events.jsonl").open("wb") as output:
        for path in events:
            output.write(path.read_bytes())
    probe = Path("results/limit_probe.jsonl")
    (raw / "limit_probe.jsonl").write_bytes(probe.read_bytes() if probe.exists() else b"")
    reviewed = Path("results/final_completion/review/grades.csv")
    grades = reviewed if reviewed.exists() else p7 / "grades.csv"
    if grades.exists():
        shutil.copyfile(grades, raw / "grades.csv")
    else:
        with (raw / "grades.csv").open("w", newline="", encoding="utf-8") as output:
            csv.DictWriter(output, fieldnames=list(Grade.model_fields)).writeheader()
    manifests = (
        sorted(p7.glob("**/manifest.json"))
        + sorted(Path("results/evidence_completion/experiments/runs").glob("*/run_manifest.json"))
        + sorted(Path("results/final_completion/experiments/runs").glob("*/run_manifest.json"))
    )
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
    for name in ["real_smoke_budget_check", "real_smoke_5000"]:
        source = Path("results") / name / "manifest.json"
        if source.exists():
            dest = raw / name / "manifest.json"
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, dest)
    for directory in ["real_smoke", "standin_evaluation", "standin_evaluation_repeat"]:
        for path in (p7 / directory).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml"}:
                target = raw / directory / path.relative_to(p7 / directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in [
        "reference_associations.json",
        "retrieval_observations.jsonl",
        "corpus_verification.json",
    ]:
        if (p7 / name).exists():
            shutil.copyfile(p7 / name, raw / name)
    if (p7 / "real_smoke/driver.py").exists():
        shutil.copyfile(p7 / "real_smoke/driver.py", raw / "real_smoke/driver.py")
    evidence = Path("results/evidence_completion")
    for name in [
        "calibration_manifest.json",
        "budget_review.json",
        "absence_search.json",
        "retrieval_review.json",
        "verification.json",
        "prompt_token_audit.json",
        "smoke_diagnosis.md",
        "reference_review.json",
        "grading_queue.csv",
        "grading_manifest.json",
        "baseline_manifest.json",
    ]:
        if (evidence / name).exists():
            shutil.copyfile(evidence / name, raw / name)
    for folder in ["reference_pages", "experiments/runs"]:
        for path in (evidence / folder).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml", ".txt", ".log"}:
                target = raw / "experiment_manifests" / path.relative_to(evidence)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in ["chatgpt", "codex"]:
        (root / "conversation_history" / name).mkdir(exist_ok=True)
    final = Path("results/final_completion")
    for name in ["evaluation", "reports"]:
        (root / name).mkdir(exist_ok=True)
    for src, dest in [
        (
            Path("eval/datasets/candidate-final/questions.jsonl"),
            root / "evaluation/questions.jsonl",
        ),
        (Path("eval/answer_keys/candidate-final.jsonl"), root / "evaluation/answer_key.jsonl"),
        (Path("reports/submission_readme.md"), root / "README.md"),
        (Path("reports/SUBMISSION_EMAIL.md"), root / "SUBMISSION_EMAIL.md"),
        (Path("reports/SUBMISSION_CHECKLIST.md"), root / "CANDIDATE_ACTIONS.md"),
        (final / "calibration_manifest.json", raw / "calibration_manifest.json"),
    ]:
        shutil.copyfile(src, dest)
    for path in Path("reports").glob("*"):
        if path.is_file():
            shutil.copyfile(path, root / "reports" / path.name)
    for directory, target in [
        (final / "review", root / "evaluation/review"),
        (final / "conversation_history", root / "conversation_history"),
        (final / "experiments", raw / "final_experiments"),
    ]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {
                ".json",
                ".jsonl",
                ".yaml",
                ".txt",
                ".csv",
                ".md",
                ".log",
            }:
                dest = target / path.relative_to(directory)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    account = final / "account_evidence"
    for path in account.glob("*"):
        (raw / "account_evidence").mkdir(exist_ok=True)
        shutil.copy2(path, raw / "account_evidence" / path.name)
        if path.suffix == ".png":
            shutil.copy2(path, raw / path.name)
    for path in final.glob("*.json"):
        shutil.copyfile(path, root / "reports" / path.name)
    files = [
        {
            "path": str(p.relative_to(root)).replace("\\", "/"),
            "size": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "artifact_type": p.suffix.lstrip(".") or "text",
            "origin": "candidate_provided"
            if p.suffix == ".png" and p.name.startswith(("limits_page_", "usage_"))
            else "assistant_review"
            if "review" in p.parts
            else "source"
            if "code" in p.relative_to(root).parts
            else "real"
            if "real_smoke" in p.parts or "connectivity" in p.name
            else "stand-in"
            if "experiment_manifests" in p.parts
            else "generated",
            "run_id": next(
                (part for part in p.parts if part.startswith(("p6-", "eval-", "real-smoke-"))), None
            ),
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
