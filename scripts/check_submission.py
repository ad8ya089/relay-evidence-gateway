"""Validate submission structure and evidence; readiness fails on absent candidate inputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

from eval.dataset import validate_dataset
from scripts.aggregate import submission_tables
from scripts.build_results_csv import read_csv
from scripts.report import headings


def member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes package")
    return path


def secret_findings(root: Path) -> list[str]:
    issues = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".env":
            issues.append(str(path.relative_to(root)) + ": prohibited environment file")
            continue
        raw = path.read_bytes()
        if (
            re.search(rb"gsk_[A-Za-z0-9_-]{20,}", raw)
            or re.search(rb"(?m)^GROQ_API_KEY=[^\s\"']+", raw)
            or re.search(rb"(?im)^\s*authorization\s*:\s*bearer\s+\S+", raw)
        ):
            issues.append(str(path.relative_to(root)) + ": credential pattern (value suppressed)")
    return issues


def pdf_issues(path: Path) -> list[str]:
    with fitz.open(path) as doc:
        issues = [] if 1 <= len(doc) <= 2 else ["PDF must contain at most two pages"]
        text = " ".join(" ".join(p.get_text() for p in doc).split())
        cursor = 0
        for heading in headings():
            position = text.find(heading, cursor)
            if position < 0:
                issues.append("Missing or misordered report heading: " + heading)
            else:
                cursor = position + len(heading)
        return issues


def validate(root: Path, *, readiness: bool = True) -> dict[str, object]:
    errors, pending = [], []
    required = [
        "writeup.pdf",
        "writeup.md",
        "results.csv",
        "README.md",
        "raw/events.jsonl",
        "raw/limit_probe.jsonl",
        "raw/run_manifest.json",
        "raw/grades.csv",
        "raw/tables.json",
        "raw/connectivity_result.json",
        "artifact_manifest.json",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append("Missing required artifact: " + name)
    if not (root / "code").is_dir():
        errors.append("Missing required code directory")
    if errors:
        return {"status": "failed", "errors": errors, "pending": pending}
    try:
        rows = read_csv(root / "results.csv")
        validate_dataset(
            root / "code/eval/datasets/candidate-v1/questions.jsonl",
            root / "code/eval/answer_keys/candidate-v1.jsonl",
        )
        source = json.loads((root / "raw/connectivity_result.json").read_text())["observed"]
        expected, _ = submission_tables(rows, source)
        actual = json.loads((root / "raw/tables.json").read_text())
        if actual != expected:
            errors.append("Tables do not match recomputed CSV/evidence values")
        errors += pdf_issues(root / "writeup.pdf")
        hashes = json.loads((root / "code/config/protected_hashes.json").read_text())
        for name, digest in hashes.items():
            path = member(root / "code", name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append("Protected hash mismatch: " + name)
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        for entry in manifest["files"]:
            path = member(root, entry["path"])
            if (
                not path.is_file()
                or path.stat().st_size != entry["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                errors.append("Artifact hash mismatch: " + entry["path"])
        if any(r.question_id and not r.answer_grade for r in rows):
            pending.append("Manual grades remain blank")
        if not list((root / "raw").glob("limits_page_*.png")):
            pending.append("Required real Limits-page screenshot is missing")
        history = [
            p
            for p in (root / "conversation_history").rglob("*")
            if p.is_file() and p.name != "README.md"
        ]
        if not history:
            pending.append("Full uncleaned conversation exports are missing")
        pending += [
            "Candidate Vaani feedback is not supplied",
            "Official real evaluation and calibrated load evidence remain incomplete",
        ]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append("Invalid submission inputs: " + type(exc).__name__)
    errors += secret_findings(root)
    return {
        "status": "failed" if errors else "blocked" if readiness and pending else "passed",
        "errors": errors,
        "pending": pending,
        "readiness_required": readiness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("relay-submission"))
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    result = validate(args.package, readiness=not args.structural_only)
    Path("results/p7/submission_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
