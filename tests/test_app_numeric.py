import pytest

from relay.app.models import Draft
from relay.app.validation import validate_evidence
from relay.retrieval.models import RetrievalHit
from tests.fixtures.backends import fake_chunk


def validate(claim: str, quote: str, source: str) -> bool:
    chunk = fake_chunk(0, source, "moshi")
    hit = RetrievalHit(**chunk.model_dump(), score=1, rank=1)
    draft = Draft(
        status="answer",
        claims=[{"text": claim, "evidence": [{"chunk_id": hit.chunk_id, "quote": quote}]}],
    )
    return validate_evidence(draft, [hit]).deterministic_passed


@pytest.mark.parametrize(
    "claim,quote,source,expected",
    [
        (
            "The theoretical latency is 160 ms.",
            "The theoretical latency is 160 ms.",
            "The theoretical latency is 160 ms.",
            True,
        ),
        (
            "The practical latency is 160 ms.",
            "The theoretical latency is 160 ms, 200 ms in practice.",
            "The theoretical latency is 160 ms, 200 ms in practice.",
            False,
        ),
        ("The latency is 160 ms.", "160 ms", "The theoretical latency is 160 ms.", False),
        ("Latency is 60 ms.", "60 ms", "Latency is 160 ms.", False),
        ("Latency is 160 s.", "Latency is 160 ms.", "Latency is 160 ms.", False),
        ("Latency is 0.16 s.", "Latency is 160 ms.", "Latency is 160 ms.", True),
        (
            "The approximate practical latency is about 200 ms.",
            "The practical latency is about 200 ms.",
            "The practical latency is about 200 ms.",
            True,
        ),
        (
            "The practical latency is 200 ms.",
            "The practical latency is about 200 ms.",
            "The practical latency is about 200 ms.",
            False,
        ),
        ("It uses two heads.", "It uses three heads.", "It uses three heads.", False),
        ("It uses two heads.", "It uses 2 heads.", "It uses 2 heads.", True),
        (
            "Latency is at most 200 ms.",
            "Latency is at least 200 ms.",
            "Latency is at least 200 ms.",
            False,
        ),
        (
            "Complexity is lower when n > d.",
            "Complexity is lower when n < d.",
            "Complexity is lower when n < d.",
            False,
        ),
        (
            "Latency is 160 ms.",
            "Latency is 160 ms when the condition holds.",
            "Latency is 160 ms when the condition holds.",
            False,
        ),
    ],
)
def test_numeric_context_and_units_fail_safely(
    claim: str, quote: str, source: str, expected: bool
) -> None:
    assert validate(claim, quote, source) == expected
