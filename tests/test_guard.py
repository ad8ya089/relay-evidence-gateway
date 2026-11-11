from concurrent.futures import ThreadPoolExecutor

import pytest

from relay.providers.guard import RealCallDenied, RealCallGuard
from relay.settings import SafetySettings


@pytest.mark.parametrize(
    "allow_real,argv,permitted",
    [
        ("0", [], False),
        ("1", [], False),
        ("0", ["--real"], False),
        ("1", ["--real"], True),
        ("1", ["--really"], False),
    ],
)
def test_requires_both_permissions(allow_real: str, argv: list[str], permitted: bool) -> None:
    guard = RealCallGuard(SafetySettings(allow_real=allow_real), argv=argv)
    if permitted:
        guard.reserve(10)
        assert guard.requests_reserved == 1
    else:
        with pytest.raises(RealCallDenied, match="AND"):
            guard.reserve(10)
        assert guard.requests_reserved == guard.tokens_reserved == 0


def test_environment_defaults_and_process_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["probe.py", "--real"])
    with pytest.raises(RealCallDenied):
        RealCallGuard().reserve(1)
    monkeypatch.setenv("RELAY_ALLOW_REAL", "1")
    monkeypatch.setenv("RELAY_REAL_TOKEN_CAP", "20")
    monkeypatch.setenv("RELAY_REAL_REQUEST_CAP", "2")
    guard = RealCallGuard()
    guard.reserve(20)
    with pytest.raises(RealCallDenied, match="TOKEN_CAP"):
        guard.reserve(1)


def test_request_cap_and_invalid_reservations() -> None:
    guard = RealCallGuard(
        SafetySettings(allow_real="1", real_request_cap=1, real_token_cap=10), argv=["--real"]
    )
    with pytest.raises(ValueError):
        guard.reserve(0)
    guard.reserve(5)
    with pytest.raises(RealCallDenied, match="REQUEST_CAP"):
        guard.reserve(1)
    assert guard.requests_reserved == 1 and guard.tokens_reserved == 5


def test_concurrent_reservations_are_atomic() -> None:
    guard = RealCallGuard(
        SafetySettings(allow_real="1", real_request_cap=10, real_token_cap=3), argv=["--real"]
    )

    def reserve() -> bool:
        try:
            guard.reserve(1)
        except RealCallDenied:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda _: reserve(), range(10)))
    assert sum(outcomes) == guard.requests_reserved == guard.tokens_reserved == 3
