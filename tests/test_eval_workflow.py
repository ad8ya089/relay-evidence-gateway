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
