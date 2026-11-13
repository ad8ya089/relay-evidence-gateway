"""Candidate questions and isolated grading references; no gold loader in model execution."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PINS = {
    "transformer": ("1706.03762", 7),
    "moshi": ("2410.00037", 1),
    "dynamic_chunking": ("2507.07955", 2),
}
QUESTIONS = Path("eval/datasets/candidate-v1/questions.jsonl")
KEY = Path("eval/answer_keys/candidate-v1.jsonl")


class CandidateQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    paper: Literal["transformer", "moshi", "dynamic_chunking"]
    arxiv_id: str
    arxiv_version: int


class AnswerKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qid: str
    answerable: bool
    paper: str
    arxiv_id: str
    arxiv_version: int
    gold_reference: str = Field(min_length=1)
    evidence_hint: str
    gold_phrases: list[str]
    grade: None = None


def load_questions(path: Path = QUESTIONS) -> list[CandidateQuestion]:
    rows = [
        CandidateQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 18 or len({q.qid for q in rows}) != 18:
        raise ValueError("Exactly 18 unique candidate question IDs required")
    for row in rows:
        if (row.arxiv_id, row.arxiv_version) != PINS[row.paper]:
            raise ValueError(f"{row.qid}: wrong pinned source/version")
    return rows


def load_key(path: Path = KEY) -> dict[str, AnswerKey]:
    rows = [
        AnswerKey.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 18 or len({r.qid for r in rows}) != 18:
        raise ValueError("Exactly 18 unique answer-key IDs required")
    return {r.qid: r for r in rows}


def validate_dataset() -> dict[str, object]:
    questions, keys = load_questions(), load_key()
    if {q.qid for q in questions} != set(keys):
        raise ValueError("Question/key IDs differ")
    for q in questions:
        k = keys[q.qid]
        if (q.paper, q.arxiv_id, q.arxiv_version) != (k.paper, k.arxiv_id, k.arxiv_version):
            raise ValueError(f"{q.qid}: key provenance mismatch")
    for paper in PINS:
        group = [k for k in keys.values() if k.paper == paper]
        if len(group) != 6 or sum(k.answerable for k in group) != 4:
            raise ValueError("Require four answerable and two unanswerable per paper")
    return {
        "questions": len(questions),
        "answerable": sum(k.answerable for k in keys.values()),
        "unanswerable": sum(not k.answerable for k in keys.values()),
        "grades": "blank",
    }


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), indent=2))
