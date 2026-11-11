import json
from pathlib import Path

import pytest

from eval.retrieval_offline import EvaluationError, evaluate, gold_hit, load_dataset
from tests.fixtures.backends import fixture_engine

FIXTURES = Path("tests/fixtures")


def test_fake_rows_rejected_by_actual_grading() -> None:
    with pytest.raises(EvaluationError, match="FAKE"):
        load_dataset(FIXTURES / "questions.jsonl", FIXTURES / "gold.jsonl")


def test_empty_dataset_actionable(tmp_path: Path) -> None:
    q, g = tmp_path / "q.jsonl", tmp_path / "g.jsonl"
    q.touch()
    g.touch()
    with pytest.raises(EvaluationError, match="Write and verify 18"):
        load_dataset(q, g)


def test_fixture_bypass_restricted_to_test_directory(tmp_path: Path) -> None:
    q, g = tmp_path / "q.jsonl", tmp_path / "g.jsonl"
    q.write_bytes((FIXTURES / "questions.jsonl").read_bytes())
    g.write_bytes((FIXTURES / "gold.jsonl").read_bytes())
    with pytest.raises(EvaluationError, match="restricted"):
        load_dataset(q, g, fixture_only=True)


async def test_fixture_metrics_and_gold_provenance() -> None:
    data = load_dataset(FIXTURES / "questions.jsonl", FIXTURES / "gold.jsonl", fixture_only=True)
    engine = fixture_engine()
    rows = await evaluate(data, engine)
    assert {row.setting for row in rows} == {"A", "B", "C", "D"}
    assert all(row.n_answerable == 2 and row.hit_at_k == 1 and row.fixture_only for row in rows)
    assert all(0 < row.mrr <= 1 and row.mean_context_tokens > 0 for row in rows)
    result = await engine.search_papers("alpha", setting="A")
    hit = next(h for h in result.hits if h.paper_id == "transformer")
    key = data.gold["FAKE-q1"]
    assert gold_hit(hit, key)
    assert not gold_hit(hit.model_copy(update={"arxiv_version": 8}), key)
    assert not gold_hit(hit.model_copy(update={"page_start": 2, "page_end": 2}), key)


def test_production_files_have_no_fake_placeholders() -> None:
    assert "FAKE" not in Path("eval/questions.jsonl").read_text(encoding="utf-8").upper()
    assert "FAKE" not in Path("eval/gold.jsonl").read_text(encoding="utf-8").upper()
    assert (
        len([json.loads(line) for line in (FIXTURES / "questions.jsonl").read_text().splitlines()])
        == 2
    )
