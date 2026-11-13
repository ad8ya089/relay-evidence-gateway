from scripts.aggregate import human_agreement, quality
from tests.fixtures.results import result


def test_ungraded_is_not_wrong_and_sources_can_be_grouped() -> None:
    rows = [
        result(record_kind="eval", question_id="A01", answerable=True),
        result(request_id="FAKE-2", record_kind="eval", question_id="A05", answerable=False),
    ]
    q = quality(rows)
    assert q["asked"] == 2 and q["ungraded"] == 2
    assert q["correct_pct"] is None and q["overall_accuracy"] is None
    assert q["answerable"] == q["unanswerable"] == 1


def test_agreement_uses_only_explicit_paired_grades() -> None:
    assert human_agreement([]) == {"n": 0, "agreement": None}
    assert human_agreement(
        [
            {"request_id": "FAKE", "round": 1, "answer_grade": "correct"},
            {"request_id": "FAKE", "round": 2, "answer_grade": "wrong"},
        ]
    ) == {"n": 1, "agreement": 0}


def test_outcome_computed_provider_count_is_reconstructed_from_attempts() -> None:
    from scripts.prepare_p7 import convert

    raw = result().model_dump()
    raw.pop("provider_calls")
    raw["calls"] = [{"dispatched": True}, {"dispatched": False}]
    raw["usage"] = {"tokens_in": 100, "tokens_out": 20, "tokens_cached": 0, "complete": True}
    assert convert(raw, {"answerable": True}).provider_calls == 1
