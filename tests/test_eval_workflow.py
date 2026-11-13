import inspect

from eval import runner
from eval.references import hit


def test_execution_does_not_load_answer_key() -> None:
    source = inspect.getsource(runner)
    assert "load_key" not in source and "gold_reference" not in source


def test_real_headroom_is_not_assumed_from_old_headers() -> None:
    plan = runner.real_plan("A", 18)
    assert plan["status"] == "blocked"
    assert plan["worst_case_tokens"] > plan["session_cap"]


def test_grounded_phrase_hit_and_unknown_are_distinct() -> None:
    assert hit(["A literal supported phrase exists"], [{"phrase": "literal supported phrase"}])
    assert hit(["unrelated"], [{"phrase": "literal supported phrase"}]) is False
    assert hit(["unrelated"], []) is None


def test_anchor_quote_and_pin_must_match_the_actual_chunk() -> None:
    import pytest

    from eval.references import validate_anchors
    from relay.retrieval.models import Chunk

    chunk = Chunk(
        paper_id="transformer",
        title="FAKE fixture",
        arxiv_id="1706.03762",
        arxiv_version=7,
        url="https://example.invalid",
        chunk_id="FAKE",
        section="FAKE",
        page_start=1,
        page_end=1,
        kind="text",
        text="A literal fixture phrase.",
        n_tokens=5,
    )
    anchor = {
        "chunk_id": "FAKE",
        "quote": "literal fixture phrase",
        "arxiv_id": "1706.03762",
        "arxiv_version": 7,
        "section": "FAKE",
        "page_start": 1,
    }
    validate_anchors({"FAKE": chunk}, [anchor])
    with pytest.raises(ValueError, match="metadata"):
        validate_anchors({"FAKE": chunk}, [anchor | {"arxiv_version": 6}])
