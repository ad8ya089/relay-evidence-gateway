import pytest

from relay.app.models import Draft
from relay.app.text import normalize
from relay.app.validation import validate_evidence
from tests.fixtures.backends import fixture_engine


async def example() -> tuple[Draft, list]:
    hits = (await fixture_engine().search_papers("alpha", setting="A")).hits
    draft = Draft(
        status="answer",
        claims=[
            {
                "text": "FAKE alpha evidence",
                "evidence": [{"chunk_id": hits[0].chunk_id, "quote": hits[0].text}],
            }
        ],
    )
    return draft, hits


async def test_membership_and_quotes_and_required_verification() -> None:
    draft, hits = await example()
    result = validate_evidence(draft, hits)
    assert result.deterministic_passed and not result.verifier_passed
    draft.claims[0].evidence[0].chunk_id = "forged"
    assert validate_evidence(draft, hits).status == "fail_final"
    draft.claims[0].evidence[0].chunk_id = hits[0].chunk_id
    draft.claims[0].evidence[0].quote = "A semantically similar but absent statement"
    assert validate_evidence(draft, hits).status == "fail_repairable"


@pytest.mark.parametrize(
    "changes",
    [
        {"arxiv_version": 999},
        {"arxiv_version": 0},
        {"section": None},
        {"page_start": 0},
        {"title": "model invented title"},
        {"page_end": 0},
    ],
)
async def test_invalid_provenance_never_passes(changes: dict[str, object]) -> None:
    draft, hits = await example()
    assert validate_evidence(draft, [hits[0].model_copy(update=changes)]).status == "fail_final"


def test_normalization_is_exact_not_semantic() -> None:
    assert (
        normalize("inter-\nnal\n evidence \u00a0 fullwidth １２")
        == "internal evidence fullwidth 12"
    )
    assert normalize("encoder-decoder") == "encoder-decoder"
    assert normalize("not supported") != normalize("supported")


async def test_long_quote_and_model_citation_are_repairable() -> None:
    draft, hits = await example()
    text = "word " * 41
    hits = [hits[0].model_copy(update={"text": text})]
    draft.claims[0].evidence[0].quote = text
    assert "quote_too_long" in validate_evidence(draft, hits).claims[0].reasons
    draft.claims[0].text = "FAKE claim [Moshi v1, §99, p.999]"
    assert "model_authored_citation" in validate_evidence(draft, hits).claims[0].reasons
