from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from relay.control.ledger import CallRecord, Ledger
from relay.providers.base import Usage
from relay.providers.guard import RealCallDenied, RealCallGuard
from relay.settings import SafetySettings


def guard(ledger: Ledger, *, cap: int = 100, requests: int = 10) -> RealCallGuard:
    return RealCallGuard(
        SafetySettings(allow_real="1", real_token_cap=cap, real_request_cap=requests),
        argv=["--real"],
        ledger=ledger,
        session_id="same-session",
    )


def attempt(index: int, model: str = "FAKE-model") -> CallRecord:
    return CallRecord(
        id=f"c{index}",
        run_id="test",
        kind="probe",
        model=model,
        status="pending",
        real=True,
        session_id="same-session",
        request_id=f"r{index}",
    )


def test_inflight_requests_across_guards_and_models_cannot_bypass_caps(tmp_path: Path) -> None:
    path = tmp_path / "usage.sqlite3"
    with Ledger(path) as first, Ledger(path) as second:
        one, two = guard(first), guard(second)
        one.reserve(80, record=attempt(1))
        with pytest.raises(RealCallDenied, match="TOKEN_CAP"):
            two.reserve(30, record=attempt(2, "other-model"))
        settlement = one.reconcile(
            "c1",
            usage=Usage(tokens_in=50, tokens_cached=30, tokens_out=10, total_tokens=60),
            status="ok",
        )
        assert settlement.charged_tokens == 30
        two.reserve(70, record=attempt(2))
        assert one.tokens_reserved == 100 and one.requests_reserved == 2


def test_unknown_failures_keep_full_charge_and_attempt_count(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        safety = guard(ledger, cap=100, requests=1)
        safety.reserve(80, record=attempt(1))
        safety.reconcile(
            "c1", usage=Usage(), status="error", status_code=429, error_kind="rate_limit_unknown"
        )
        assert safety.tokens_reserved == 80
        assert ledger.session_usage("same-session").unknown_usage_count == 1
        with pytest.raises(RealCallDenied, match="REQUEST_CAP"):
            safety.reserve(1, record=attempt(2))


def test_atomic_reservations_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "usage.sqlite3"
    with Ledger(path) as first, Ledger(path) as second:
        guards = [guard(first, cap=3), guard(second, cap=3)]

        def reserve(index: int) -> bool:
            try:
                guards[index % 2].reserve(1, record=attempt(index))
            except RealCallDenied:
                return False
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert sum(pool.map(reserve, range(10))) == 3
        assert first.session_usage("same-session").attempts == 3


def test_unbounded_historical_usage_blocks_new_real_calls(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        ledger.record_call(attempt(1).model_copy(update={"status": "error"}))
        with pytest.raises(RealCallDenied, match="unbounded"):
            guard(ledger).reserve(1, record=attempt(2))


def test_underestimated_actual_usage_exposes_exceeded_cap(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        safety = guard(ledger)
        safety.reserve(50, record=attempt(1))
        settlement = safety.reconcile(
            "c1",
            usage=Usage(tokens_in=100, tokens_out=10, tokens_cached=0, total_tokens=110),
            status="ok",
        )
        assert settlement.cap_exceeded
        with pytest.raises(RealCallDenied):
            safety.reserve(1, record=attempt(2))


def test_partial_usage_cannot_hide_an_overrun_behind_missing_cached_tokens(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        safety = guard(ledger)
        safety.reserve(80, record=attempt(1))
        settlement = safety.reconcile(
            "c1", usage=Usage(tokens_in=100, tokens_out=20, total_tokens=120), status="ok"
        )
        row = ledger.get_call("c1")
        assert row.tokens_cached is None and row.counted_tokens is None
        assert settlement.usage_unknown and settlement.charged_tokens == 120
        assert settlement.cap_exceeded
