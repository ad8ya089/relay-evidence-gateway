from datetime import UTC, datetime

from relay.obs.clock import FakeClock
from standin.profile import Profile, load_profile


def clock() -> FakeClock:
    return FakeClock(utc=datetime(2026, 11, 12, 23, 59, 30, tzinfo=UTC))


def profile(**overrides: object) -> Profile:
    data = load_profile().model_dump()
    for name in (
        "server_error_prob",
        "timeout_prob",
        "disconnect_prob",
        "slow_success_prob",
        "verify_fail_prob",
    ):
        data["faults"][name]["value"] = 0
    for path, value in overrides.items():
        parts = path.split("__")
        target = data
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]]["value"] = value
    return Profile.model_validate(data)


async def advance_sleep(seconds: float) -> None:
    raise AssertionError("Tests must inject a clock-bound sleep")
