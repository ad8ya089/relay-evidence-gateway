import pytest

from relay.app.gate import EvidenceConflict, broadening_plan, evaluate, merge_results
from relay.app.interpret import interpret
from relay.retrieval.models import RetrievalResult
from tests.fixtures.backends import fixture_engine


async def test_empty_weak_sufficient_are_distinct_and_errors_do_not_refuse() -> None:
    result = await fixture_engine().search_papers("alpha", setting="A")
    assert evaluate(result, interpret("alpha")).status == "sufficient"
    assert evaluate(result, interpret("unmatched context")).status == "weak"
    empty = result.model_copy(update={"hits": [], "status": "empty"})
    assert evaluate(empty, interpret("alpha")).status == "empty"
    with pytest.raises(ValueError):
        evaluate(empty.model_copy(update={"status": "timeout"}), interpret("alpha"))


async def test_broadening_is_meaningful_bounded_and_preserves_explicit_scope() -> None:
    result = await fixture_engine().search_papers("alpha", setting="A")
    text = "In the Moshi paper, explain alpha latency"
    i = interpret(text)
    one = broadening_plan(text, i, result, 1, [text])
    two = broadening_plan(text, i, result, 2, [text, one.query])
    assert one.query != two.query and two.numeric
    assert one.paper_filter == two.paper_filter == "moshi"
    assert broadening_plan(text, i, result, 3, [text]) is None
    assert broadening_plan(text, i, result, 2, [two.query]) is None


async def test_hint_can_be_removed_without_losing_other_papers() -> None:
    engine = fixture_engine()
    a = await engine.search_papers("alpha", setting="A")
    b = await engine.search_papers("beta", setting="A")
    merged = merge_results(a, b)
    assert [h.paper_id for h in merged.hits] == ["transformer", "moshi"]
    query = "Moshi alpha method"
    plan = broadening_plan(query, interpret(query), merged, 2, [query])
    assert plan.paper_filter is None and plan.paper_hint is None
    assert "moshi" not in plan.query.lower()
    bad = b.model_copy(update={"hits": [a.hits[0].model_copy(update={"text": "changed"})]})
    with pytest.raises(EvidenceConflict):
        merge_results(a, bad)


def test_unknown_domain_does_not_add_unrelated_section_terms() -> None:
    empty = RetrievalResult(status="empty", setting="A", method="bm25", query_used="zqxvbn")
    i = interpret("zqxvbn")
    assert broadening_plan("zqxvbn", i, empty, 1, ["zqxvbn"]) is None
    assert broadening_plan("zqxvbn", i, empty, 2, ["zqxvbn"]) is None
