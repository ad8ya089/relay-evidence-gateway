from pathlib import Path

import pytest

from loadgen.scenarios import catalog, prerequisite
from loadgen.settings import load_settings
from loadgen.workload import PrerequisiteError


def test_official_shapes_and_retry_pair_are_explicit() -> None:
    order, scenarios = catalog()
    assert len(order) == 9
    assert scenarios["steady"].duration_s == 150
    assert [s.multiplier for s in scenarios["burst"].arrivals.segments] == [0.5, 4, 0.5]
    assert scenarios["retry_naive"].seed == scenarios["retry_bounded"].seed
    assert scenarios["retry_naive"].faults == scenarios["retry_bounded"].faults
    assert scenarios["burst_cache_on"].sampling == "zipf"
    assert scenarios["minute_limit_halved"].admin_events[0].offset_s == 40


def test_missing_official_prerequisites_block_and_fixture_is_explicit(tmp_path: Path) -> None:
    _, scenarios = catalog()
    empty = tmp_path / "human.jsonl"
    empty.write_text("")
    official = scenarios["steady"].model_copy(update={"workload": empty})
    with pytest.raises(PrerequisiteError, match="human-authored"):
        prerequisite(official, load_settings(), tmp_path / "missing.json")
    workload, profile, capacity = prerequisite(
        scenarios["development_smoke"], load_settings(), tmp_path / "missing"
    )
    assert workload and profile.name == "development" and capacity is None


def test_invalid_admin_order_and_sampling_fail_validation() -> None:
    scenario = catalog()[1]["steady"].model_dump()
    from loadgen.scenarios import Scenario

    with pytest.raises(ValueError, match="Cache ON"):
        Scenario.model_validate(scenario | {"sampling": "zipf"})
    with pytest.raises(ValueError, match="deterministic"):
        Scenario.model_validate(
            scenario
            | {
                "admin_events": [
                    {"offset_s": 20, "action": "reset"},
                    {"offset_s": 10, "action": "reset"},
                ]
            }
        )
