"""Explicit human grading only. Display references here, never in model execution."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eval.dataset import load_key


class Grade(BaseModel):
    request_id: str = Field(min_length=1)
    question_id: str
    retrieval_setting: Literal["A", "B", "C"]
    answer_grade: Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]
    failure_stage: Literal[
        "",
        "retrieval_miss",
        "model_misread",
        "validator_false_reject",
        "validator_miss",
        "refusal_threshold",
        "budget",
    ] = ""
    rationale: str = Field(default="", max_length=500)
    human_confirmed: Literal[True]
    round: int = Field(default=1, ge=1)


def save_grade(grade: Grade, path: Path) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    if grade.answer_grade in {"wrong", "refused_wrongly"} and not grade.failure_stage:
        raise ValueError("A failed answer requires an explicit failure stage")
    existing = (
        list(csv.DictReader(path.open(encoding="utf-8", newline=""))) if path.exists() else []
    )
    if any(
        r["request_id"] == grade.request_id and int(r["round"]) == grade.round for r in existing
    ):
        raise ValueError("Request already graded in this round; use a new regrade round")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(Grade.model_fields))
        if needs_header:
            writer.writeheader()
        writer.writerow(grade.model_dump())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, nargs="?")
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--grades", type=Path, default=Path("results/p7/grades.csv"))
    parser.add_argument("--request-id")
    parser.add_argument("--regrade", type=int)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    if args.agreement:
        from scripts.aggregate import human_agreement

        if args.grades.exists():
            with args.grades.open(encoding="utf-8", newline="") as source:
                grades = [Grade.model_validate(row).model_dump() for row in csv.DictReader(source)]
        else:
            grades = []
        print(json.dumps(human_agreement(grades), indent=2))
        return 0
    if args.rows is None:
        parser.error("Evaluation rows are required unless --agreement is selected")
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
    if args.request_id:
        rows = [r for r in rows if r["request_id"] == args.request_id]
    if args.regrade:
        rows = random.Random(args.seed).sample(rows, min(args.regrade, len(rows)))
    keys = load_key()
    for row in rows:
        key = keys[row["question_id"]]
        print(
            json.dumps(
                {
                    "question": row["question"],
                    "answerable": key.answerable,
                    "answer": row["answer"],
                    "status": row["status"],
                    "citations": row["citations"],
                    "gold_reference": key.gold_reference,
                    "expected_evidence": key.evidence_hint,
                    "validation": row.get("validation"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        chunk_path = Path("data/chunks") / f"{row['retrieval_setting']}.jsonl"
        ids = set(row.get("supplied_chunk_ids", []))
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            if chunk["chunk_id"] in ids:
                print(json.dumps(chunk, ensure_ascii=False))
        selected = input("Select explicit human grade, or Enter to leave blank: ").strip()
        if not selected:
            continue
        stage = input("Failure stage (blank for successful answer/refusal): ").strip()
        rationale = input("Optional rationale: ").strip()
        grade = Grade(
            request_id=row["request_id"],
            question_id=row["question_id"],
            retrieval_setting=row["retrieval_setting"],
            answer_grade=selected,
            failure_stage=stage,
            rationale=rationale,
            human_confirmed=True,
            round=args.round,
        )
        save_grade(grade, args.grades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
