import pytest

from loadgen.arrivals import ArrivalConfig, offsets, plan
from loadgen.settings import load_settings
from loadgen.workload import PrerequisiteError, questions


@pytest.mark.parametrize("kind", ["constant", "step", "burst"])
def test_complete_arrival_plan_has_no_response_dependency(kind: str) -> None:
    config = ArrivalConfig(
        kind=kind,
        segments=[
            {"duration_s": 2, "rate_per_minute": 120},
            {"duration_s": 1, "rate_per_minute": 240},
        ],
    )
    assert offsets(config, 20) == [
        0,
        500000000,
        1000000000,
        1500000000,
        2000000000,
        2250000000,
        2500000000,
        2750000000,
    ]


def test_ramp_one_shot_bounds_and_seeded_sampling() -> None:
    assert offsets(ArrivalConfig(kind="one_shot", count=6), 6) == [0] * 6
    ramp = ArrivalConfig(
        kind="ramp", segments=[{"duration_s": 2, "rate_per_minute": 60, "end_rate_per_minute": 180}]
    )
    times = offsets(ramp, 20)
    assert len(times) == 4 and times == sorted(times) and times[-1] < 2_000_000_000
    kwargs = dict(run_id="FAKE", seed=7, setting="A", sampling="zipf", maximum=5000, exponent=1.2)
    config = ArrivalConfig(kind="one_shot", count=1000)
    first = plan(config, ["q1", "q2", "q3"], **kwargs)
    assert first == plan(config, ["q1", "q2", "q3"], **kwargs)
    assert len({r.request_id for r in first}) == 1000
    assert sum(r.question_id == "q1" for r in first) > sum(r.question_id == "q3" for r in first)
    with pytest.raises(ValueError):
        offsets(config, 10)


def test_official_empty_workload_is_blocked_without_writing_questions() -> None:
    from pathlib import Path

    with pytest.raises(PrerequisiteError):
        questions(Path("eval/questions.jsonl"), load_settings())
