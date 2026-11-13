import pytest
from pydantic import ValidationError

from relay.app.models import Draft, Question, Verification
from relay.app.settings import pipeline_settings


@pytest.mark.parametrize("text", ["", "  ", "x" * 4097])
def test_question_bounds(text: str) -> None:
    with pytest.raises(ValidationError):
        Question(text=text)


def test_draft_matches_p3_and_rejects_fabricated_model_metadata() -> None:
    claim = {"text": "A", "evidence": [{"chunk_id": "supplied", "quote": "A"}]}
    assert Draft(status="answer", claims=[claim]).claims[0].evidence[0].quote == "A"
    for invalid in (
        {"status": "answer", "claims": []},
        {"status": "insufficient_evidence", "claims": [claim]},
        {"status": "made_up", "claims": []},
        {"status": "answer", "claims": [claim | {"page": 7}]},
    ):
        with pytest.raises(ValidationError):
            Draft.model_validate(invalid)


def test_verifier_schema_and_configuration_limits() -> None:
    assert Verification(verdicts=[{"i": 0, "v": "overstated", "why": "Too strong"}])
    for value in ("supported-ish", "", None):
        with pytest.raises(ValidationError):
            Verification(verdicts=[{"i": 0, "v": value, "why": ""}])
    assert pipeline_settings().deadline_ms == 8000
    assert pipeline_settings().repair_max == 1
