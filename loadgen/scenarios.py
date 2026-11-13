"""Validated experiment definitions and strict official-run prerequisites."""

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from eval.retrieval_offline import Question
from loadgen.arrivals import ArrivalConfig
from loadgen.settings import LoadSettings, Model
from loadgen.workload import PrerequisiteError, questions
from standin.profile import Profile, load_profile


class AdminEvent(Model):
    offset_s: float = Field(ge=0)
    action: Literal["halve_minute_limits", "limits", "profile", "reset", "faults"]
    factor: float | None = Field(default=None, gt=0, lt=1)
    values: dict[str, object] = Field(default_factory=dict)


class Scenario(Model):
    name: str
    label: str
    profile: Literal["development", "scaled", "free_exact"]
    seed: int
    setting: Literal["A", "B", "C"]
    deadline_ms: int = Field(gt=0)
    cache: bool
    sampling: Literal["uniform", "zipf"]
    retry_policy: Literal["bounded", "naive_s7a"]
    faults: dict[str, float]
    admin_events: list[AdminEvent]
    policy_overrides: dict[str, object]
    fixture_only: bool
    workload: Path
    align_policy_limits: bool
    duration_s: float = Field(ge=0)
    expected_requests: str
    arrivals: ArrivalConfig
    daily_budget_fraction: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def consistent(self) -> "Scenario":
        if self.cache != (self.sampling == "zipf"):
            raise ValueError("Cache ON uses Zipf; cache OFF uses uniform")
        duration = sum(s.duration_s for s in self.arrivals.segments)
        if abs(self.duration_s - duration) > 1e-9:
            raise ValueError("Scenario duration differs from arrival duration")
        if any(e.offset_s > self.duration_s for e in self.admin_events):
            raise ValueError("Admin event is outside the scenario")
        if [e.offset_s for e in self.admin_events] != sorted(e.offset_s for e in self.admin_events):
            raise ValueError("Admin events must be in deterministic time order")
        return self


def catalog(path: Path = Path("config/scenarios.yaml")) -> tuple[list[str], dict[str, Scenario]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios = {
        name: Scenario.model_validate(data["defaults"] | values | {"name": name})
        for name, values in data["scenarios"].items()
    }
    if len(set(data["order"])) != len(data["order"]) or any(
        n not in scenarios for n in data["order"]
    ):
        raise ValueError("Invalid scenario order")
    return data["order"], scenarios


def prerequisite(
    scenario: Scenario, settings: LoadSettings, capacity_path: Path
) -> tuple[dict[str, Question], Profile, dict[str, object] | None]:
    failures = []
    workload, profile, capacity = {}, None, None
    try:
        workload = questions(scenario.workload, settings, fixture_only=scenario.fixture_only)
    except (ValueError, OSError) as exc:
        failures.append(str(exc))
    try:
        profile = load_profile(scenario.profile)
    except ValueError as exc:
        failures.append(str(exc))
    needs_capacity = any(s.multiplier is not None for s in scenario.arrivals.segments)
    if needs_capacity:
        try:
            capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
            if (
                capacity.get("kind") != "S"
                or capacity.get("fixture_only")
                or capacity.get("profile") != scenario.profile
                or capacity.get("setting") != scenario.setting
                or capacity.get("deadline_ms") != scenario.deadline_ms
                or not capacity.get("value")
                or profile
                and capacity.get("profile_snapshot") != profile.model_dump()
            ):
                raise ValueError(
                    "Capacity report is incompatible with this profile/model/setting/deadline"
                )
        except (ValueError, OSError) as exc:
            failures.append(f"Run loadgen capacity after account/workload setup: {exc}")
    if (
        scenario.daily_budget_fraction
        and capacity
        and not capacity.get("mean_counted_tokens_per_answer")
    ):
        failures.append(
            "S5 needs measured counted tokens/answer from compatible capacity observations"
        )
    if failures:
        raise PrerequisiteError("; ".join(failures))
    return workload, profile, capacity
