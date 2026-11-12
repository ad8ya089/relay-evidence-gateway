import json

import pytest

from relay.providers.base import Message
from standin.generation import Draft, InvalidInput, Verification, generate


def messages(kind: str, text: str) -> list[Message]:
    return [
        Message(role="system", content=f"RELAY_{kind.upper()}_V1"),
        Message(role="user", content=text),
    ]


def test_draft_is_grounded_multiline_and_bounded() -> None:
    content = (
        "First line with 160 ms theoretical latency;\n"
        "practical latency is 200 ms. Another sentence."
    )
    draft = Draft.model_validate_json(
        generate(messages("draft", f"<<chunk:exact-id>>{content}<<end>>"))
    )
    evidence = draft.claims[0].evidence[0]
    assert evidence.chunk_id == "exact-id" and evidence.quote in content
    assert draft.claims[0].text == evidence.quote
    assert "practical" in evidence.quote
    long = " ".join(f"word{i}" for i in range(100))
    bounded = Draft.model_validate_json(generate(messages("draft", f"<<chunk:long>>{long}<<end>>")))
    assert len(bounded.claims[0].evidence[0].quote.split()) == 40


@pytest.mark.parametrize("text", ["No evidence", "<<chunk:empty>> \n <<end>>"])
def test_no_evidence_explicitly_refuses(text: str) -> None:
    draft = Draft.model_validate_json(generate(messages("draft", text)))
    assert draft.status == "insufficient_evidence" and not draft.claims


@pytest.mark.parametrize("text", ["<<chunk:missing>>x", "<<chunk:id>>a<<end>><<chunk:id>>b<<end>>"])
def test_malformed_chunks_are_explicit_errors(text: str) -> None:
    with pytest.raises(InvalidInput):
        generate(messages("draft", text))


def test_verifier_indices_and_predictable_failure() -> None:
    text = json.dumps(
        [{"i": 3, "claim": "A", "quotes": ["A"]}, {"i": 9, "claim": "B", "quotes": ["B"]}]
    )
    passed = Verification.model_validate_json(generate(messages("verify", text)))
    assert [(v.i, v.v) for v in passed.verdicts] == [(3, "supported"), (9, "supported")]
    failed = Verification.model_validate_json(generate(messages("verify", text), verify_fail=True))
    assert [(v.i, v.v) for v in failed.verdicts] == [(3, "unsupported"), (9, "supported")]


@pytest.mark.parametrize(
    "text",
    [
        "garbage",
        "[]",
        '[{"i":0,"claim":"A","quotes":[]}]',
        '[{"i":0,"claim":"A","quotes":[""]}]',
        '[{"i":0,"claim":" ","quotes":["A"]}]',
    ],
)
def test_malformed_verification_never_passes(text: str) -> None:
    with pytest.raises(InvalidInput):
        generate(messages("verify", text))


def test_markers_are_static_only_and_generic_is_deterministic() -> None:
    assert json.loads(generate([Message(role="user", content="RELAY_VERIFY_V1")])) == {
        "answer": "ok"
    }
    with pytest.raises(InvalidInput):
        generate([Message(role="developer", content="RELAY_VERIFY_V1 RELAY_DRAFT_V1")])
