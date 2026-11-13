"""Validate submission structure and evidence; readiness fails on absent candidate inputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

from eval.dataset import load_key, load_questions, validate_dataset
from scripts.aggregate import submission_tables
from scripts.build_results_csv import call_metrics, events_by_request, json_rows, read_csv
from scripts.report import headings
from scripts.result_models import ResultRow


def member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes package")
    return path


def evaluation_issues(rows: list[ResultRow], root: Path) -> list[str]:
    questions = {
        q.qid: q for q in load_questions(root / "code/eval/datasets/candidate-v1/questions.jsonl")
    }
    keys = load_key(root / "code/eval/answer_keys/candidate-v1.jsonl")
    seen = set()
    issues = []
    for row in rows:
        if row.record_kind != "eval" or row.question_id is None:
            continue
        identity = (row.run_id, row.retrieval_setting, row.question_id)
        if identity in seen or row.question_id not in questions:
            issues.append(row.request_id + ": duplicate or unknown candidate logical question")
            continue
        seen.add(identity)
        if (
            row.paper_id != questions[row.question_id].paper
            or row.answerable != keys[row.question_id].answerable
        ):
            issues.append(row.request_id + ": candidate paper/answerability mismatch")
    return issues


def secret_findings(root: Path) -> list[str]:
    issues = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".env":
            issues.append(str(path.relative_to(root)) + ": prohibited environment file")
            continue
        raw = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            try:
                with fitz.open(path) as document:
                    raw += "\n".join(page.get_text() for page in document).encode()
            except (ValueError, RuntimeError):
                issues.append(str(path.relative_to(root)) + ": unreadable PDF")
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
        errors += evaluation_issues(rows, root)
        validate_dataset(
            root / "code/eval/datasets/candidate-v1/questions.jsonl",
            root / "code/eval/answer_keys/candidate-v1.jsonl",
        )
        source = json.loads((root / "raw/connectivity_result.json").read_text())["observed"]
        expected, diagnostics = submission_tables(rows, source)
        actual = json.loads((root / "raw/tables.json").read_text())
        if actual != expected:
            errors.append("Tables do not match recomputed CSV/evidence values")
        for outcome in actual["Table 2b: Outcomes"]["rows"]:
            if outcome[1] != "not measured":
                if abs(sum(float(v) for v in outcome[1:5]) - 100) > 0.2:
                    errors.append("Table 2b outcomes do not reconcile: " + outcome[0])
        errors += pdf_issues(root / "writeup.pdf")
        old_path = root / "raw/p6_results.csv"
        anchor = json.loads((root / "raw/historical_hashes.json").read_text())
        if hashlib.sha256(old_path.read_bytes()).hexdigest() != anchor["results/results.csv"]:
            errors.append("Original P6 CSV bytes changed")
        old_rows = read_csv(old_path)
        indexed = {r.request_id: r for r in rows}
        if len(old_rows) != 84 or any(
            r.request_id not in indexed or r != indexed[r.request_id] for r in old_rows
        ):
            errors.append("Historical P6 row set changed or missing")
        grouped = events_by_request(json_rows([root / "raw/events.jsonl"]))
        for row in rows:
            measured = call_metrics(row.request_id, grouped.get(row.request_id, []))
            if measured and row.provider_calls != measured["provider_calls"]:
                errors.append(row.request_id + ": provider accounting differs from raw calls")
            for field in ("tokens_in", "tokens_out", "tokens_cached"):
                if field in measured and getattr(row, field) != measured[field]:
                    errors.append(row.request_id + ": usage differs from raw calls")
        hashes = json.loads((root / "code/config/protected_hashes.json").read_text())
        for name, digest in hashes.items():
            path = member(root / "code", name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append("Protected hash mismatch: " + name)
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        declared = [entry["path"] for entry in manifest["files"]]
        actual_paths = {
            str(p.relative_to(root)).replace("\\", "/")
            for p in root.rglob("*")
            if p.is_file() and p.name != "artifact_manifest.json"
        }
        if len(set(declared)) != len(declared) or set(declared) != actual_paths:
            errors.append("Artifact manifest omits, duplicates or invents files")
        for entry in manifest["files"]:
            path = member(root, entry["path"])
            if (
                not path.is_file()
                or path.stat().st_size != entry["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                errors.append("Artifact hash mismatch: " + entry["path"])
        selected = {
            diagnostics.get(f"evaluation/{source}/selected_run") for source in ["real", "stand-in"]
        }
        if any(r.question_id and r.run_id in selected and not r.answer_grade for r in rows):
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
        if "CANDIDATE INPUT REQUIRED" in (root / "writeup.md").read_text(encoding="utf-8"):
            pending.append("Candidate Vaani feedback is not supplied")
        if not all(
            len(
                {
                    r.question_id
                    for r in rows
                    if r.source == "real" and r.question_id and r.retrieval_setting == s
                }
            )
            == 18
            for s in ["A", "B", "C"]
        ):
            pending.append("Official real evaluation evidence remains incomplete")
        if not all(
            any(r.scenario == s and not r.fixture_only for r in rows)
            for s in [
                "steady",
                "burst",
                "sustained_overload",
                "minute_limit_halved",
                "daily_exhausted",
            ]
        ):
            pending.append("Official calibrated load evidence remains incomplete")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append("Invalid submission inputs: " + type(exc).__name__)
    errors += secret_findings(root)
    return {
        "status": "failed" if errors else "blocked" if readiness and pending else "passed",
        "errors": errors,
        "pending": pending,
        "readiness_required": readiness,
        "evidence_states": {
            "structure": "PASS" if not errors else "BLOCKED",
            "candidate_inputs": "CANDIDATE ACTION REQUIRED" if pending else "PASS",
            "real_evaluation_and_full_calibration": "NOT MEASURED" if pending else "PASS",
        },
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
