"""Zero-LLM retrieval diagnostics. Human-written actual evaluation data only."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from relay.retrieval.models import RetrievalHit
from relay.retrieval.search import RetrievalEngine
from relay.settings import PaperId, Setting


class EvaluationError(ValueError):
    """An unsafe/incomplete dataset or failed retrieval cannot be graded."""


class Question(BaseModel):
    qid: str
    question: str
    paper: PaperId | None
    answerable: bool
    type: Literal["number", "explain", "compare", "limitation", "absent"]


class Gold(BaseModel):
    qid: str
    gold_claims: list[str]
    gold_phrases: list[str]
    paper: PaperId | None
    version: int | None
    section: str | None
    page: int | None = Field(ge=1)
    must_refuse: bool
    notes: str


class Dataset(BaseModel):
    questions: list[Question]
    gold: dict[str, Gold]
    fixture_only: bool = False


class Metrics(BaseModel):
    setting: Setting
    n_answerable: int
    hit_at_k: float
    mrr: float
    wrong_paper_at_1: float
    mean_context_tokens: float
    phrase_rule: Literal["any"] = "any"
    rerank_methods: list[str]
    diagnostics: list[str]
    fixture_only: bool


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise EvaluationError(f"Missing {path}; write your verified human-authored rows first.")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_dataset(questions: Path, gold: Path, *, fixture_only: bool = False) -> Dataset:
    qrows, grows = _rows(questions), _rows(gold)
    if not qrows or not grows:
        raise EvaluationError(
            "Evaluation dataset is empty. Write and verify 18 questions in eval/questions.jsonl "
            "and matching gold keys in eval/gold.jsonl (12 answerable, 6 unanswerable). "
            "For synthetic plumbing only: make retr-eval EVAL_ARGS=--test-fixtures."
        )
    if fixture_only:
        fixture_dir = (Path(__file__).resolve().parents[1] / "tests/fixtures").resolve()
        if not all(path.resolve().is_relative_to(fixture_dir) for path in (questions, gold)):
            raise EvaluationError(
                "Fixture mode is restricted to tests/fixtures; cannot grade real data"
            )
    elif re.search(r"fake", json.dumps(qrows + grows), re.I):
        raise EvaluationError("FAKE placeholder detected: actual grading refuses all FAKE rows")
    parsed_q = [Question.model_validate(row) for row in qrows]
    parsed_g = [Gold.model_validate(row) for row in grows]
    qids = [q.qid for q in parsed_q]
    gids = [g.qid for g in parsed_g]
    if len(set(qids)) != len(qids) or len(set(gids)) != len(gids) or set(qids) != set(gids):
        raise EvaluationError("Questions and gold must have unique, identical qid sets")
    by_id = {g.qid: g for g in parsed_g}
    for question in parsed_q:
        key = by_id[question.qid]
        if question.answerable == key.must_refuse or question.paper != key.paper:
            raise EvaluationError(f"Question/gold mismatch: {question.qid}")
        if question.answerable and (
            not key.gold_phrases
            or any(not phrase.strip() for phrase in key.gold_phrases)
            or not key.gold_claims
            or not key.version
            or not key.page
            or not key.section
            or not key.paper
        ):
            raise EvaluationError(f"Incomplete answerable gold metadata: {question.qid}")
    if not fixture_only:
        if len(parsed_q) != 18 or sum(q.answerable for q in parsed_q) != 12:
            raise EvaluationError(
                "Actual dataset must contain 18 questions: 12 answerable + 6 absent"
            )
        for paper in ("transformer", "moshi", "dynamic_chunking"):
            if sum(q.answerable and q.paper == paper for q in parsed_q) != 4:
                raise EvaluationError(f"Actual dataset needs 4 answerable questions for {paper}")
    return Dataset(questions=parsed_q, gold=by_id, fixture_only=fixture_only)


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def gold_hit(hit: RetrievalHit, key: Gold) -> bool:
    return (
        hit.paper_id == key.paper
        and hit.arxiv_version == key.version
        and key.page is not None
        and hit.page_start <= key.page <= hit.page_end
        and key.section is not None
        and _normal(key.section) in _normal(hit.section or "")
        and any(_normal(phrase) in _normal(hit.text) for phrase in key.gold_phrases)
    )


async def evaluate(dataset: Dataset, engine: RetrievalEngine) -> list[Metrics]:
    questions = [q for q in dataset.questions if q.answerable]
    if not questions:
        raise EvaluationError("No answerable questions available for retrieval diagnostics")
    rows: list[Metrics] = []
    for setting in engine.config.settings:
        hits = reciprocal = wrong = tokens = 0.0
        rerank_methods: set[str] = set()
        diagnostics: set[str] = set()
        for question in questions:
            key = dataset.gold[question.qid]
            # Do not leak the gold paper as a hard filter or hint into retrieval.
            result = await engine.search_papers(question.question, setting=setting)
            if result.status in {"error", "timeout"}:
                raise EvaluationError(
                    f"{setting}/{question.qid}: retrieval {result.status}: {result.diagnostics}"
                )
            rank = next((hit.rank for hit in result.hits if gold_hit(hit, key)), None)
            hits += rank is not None
            reciprocal += 1 / rank if rank is not None else 0
            wrong += bool(result.hits and result.hits[0].paper_id != key.paper)
            tokens += sum(hit.n_tokens for hit in result.hits)
            rerank_methods.add(result.rerank_method)
            diagnostics.update(result.diagnostics)
        count = len(questions)
        rows.append(
            Metrics(
                setting=setting,
                n_answerable=count,
                hit_at_k=hits / count,
                mrr=reciprocal / count,
                wrong_paper_at_1=wrong / count,
                mean_context_tokens=tokens / count,
                rerank_methods=sorted(rerank_methods),
                diagnostics=sorted(diagnostics),
                fixture_only=dataset.fixture_only,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-fixtures", action="store_true")
    args = parser.parse_args()
    try:
        if args.test_fixtures:
            from tests.fixtures.backends import fixture_engine

            fixture_dir = Path("tests/fixtures")
            dataset = load_dataset(
                fixture_dir / "questions.jsonl", fixture_dir / "gold.jsonl", fixture_only=True
            )
            engine = fixture_engine()
            print("FAKE TEST FIXTURES ONLY — synthetic backends; not paper retrieval measurements.")
        else:
            dataset = load_dataset(args.questions, args.gold)
            engine = RetrievalEngine(data_dir=args.data_dir)
        for metrics in asyncio.run(evaluate(dataset, engine)):
            print(metrics.model_dump_json())
        print(
            "Hit rule: any gold phrase, matching paper/version/section/PDF page. "
            "Denominator: answerable questions; empty top hits are not wrong-paper hits."
        )
    except (EvaluationError, ValueError, OSError) as exc:
        print(f"Evaluation refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
