"""Pinned-passage boundary regression; original rejected quote was not retained."""

import json
from pathlib import Path

from relay.app.models import Draft
from relay.app.validation import validate_evidence
from relay.retrieval.models import RetrievalHit


def test_pinned_attention_literal_passes_but_altered_quote_and_mapping_fail() -> None:
    chunks = [
        json.loads(line)
        for line in Path("data/chunks/A.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chunk = next(c for c in chunks if c["chunk_id"] == "transformer-v7-p5-c20")
    hit = RetrievalHit.model_validate(chunk | {"score": 1, "rank": 1})
    draft = Draft(
        status="answer",
        claims=[
            {
                "text": "The queries come from the previous decoder layer.",
                "evidence": [
                    {
                        "chunk_id": hit.chunk_id,
                        "quote": "the queries come from the previous decoder layer",
                    }
                ],
            }
        ],
    )
    assert validate_evidence(draft, [hit]).deterministic_passed
    assert not validate_evidence(draft, [hit]).verifier_passed
    draft.claims[0].evidence[0].quote = "the queries come from the previous encoder layer"
    assert validate_evidence(draft, [hit]).failure_reason == "quote_not_substring"
    draft.claims[0].evidence[0].chunk_id = "transformer-v7-p99-c999"
    assert validate_evidence(draft, [hit]).failure_reason == "unknown_chunk_id"
