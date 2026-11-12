from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from relay.control.ledger import CallRecord, DuplicateCallError, Ledger, LedgerError
from relay.obs.clock import FakeClock


def call(index: int, **changes: object) -> CallRecord:
    fields: dict[str, object] = {
        "id": f"c{index}",
        "run_id": "test",
        "session_id": "session",
        "request_id": "r1",
        "kind": "draft",
        "model": "FAKE-model",
        "status": "ok",
        "real": True,
        "tokens_in": 100,
        "tokens_out": 20,
        "tokens_cached": 30,
        "total_tokens": 120,
    }
    return CallRecord.model_validate(fields | changes)


def test_cached_accounting_models_and_attempts(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        ledger.record_call(call(1))
        ledger.record_call(call(2, model="other"))
        ledger.record_call(
            call(
                3,
                status="error",
                status_code=429,
                tokens_in=None,
                tokens_out=None,
                tokens_cached=None,
                total_tokens=None,
                reserved_tokens=200,
            )
        )
        assert ledger.tokens_used_today("FAKE-model") == 90
        assert ledger.requests_used_today("FAKE-model") == 2
        assert ledger.tokens_used_today("other") == 90
        row = ledger.get_call("c3")
        assert row.tokens_in is None and row.total_tokens is None and row.counted_tokens is None
        summary = ledger.session_usage("session")
        assert summary.attempts == 3 and summary.known_counted_tokens == 180
        assert summary.reported_total_tokens == 240 and summary.unknown_usage_count == 1
        assert summary.guard_charged_tokens == 380


def test_utc_midnight_boundaries(tmp_path: Path) -> None:
    clock = FakeClock(utc=datetime.fromisoformat("2026-11-11T23:59:59.999000+00:00"))
    with Ledger(tmp_path / "usage.sqlite3", clock=clock) as ledger:
        ledger.record_call(call(1))
        clock.advance_ms(1)
        assert ledger.requests_used_today("FAKE-model") == 0
        ledger.record_call(call(2))
        ledger.record_call(call(3, ts_utc=datetime.fromisoformat("2026-11-12T05:29:00+05:30")))
        assert ledger.requests_used_today("FAKE-model") == 1
        assert ledger.get_call("c1").ts_utc == datetime.fromisoformat("2026-11-11T23:59:59.999000Z")


def test_duplicate_ids_and_finalisation_are_explicit(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        ledger.record_call(
            call(
                1,
                status="pending",
                tokens_in=None,
                tokens_out=None,
                tokens_cached=None,
                total_tokens=None,
            )
        )
        with pytest.raises(DuplicateCallError):
            ledger.record_call(call(1))
        ledger.finish_call("c1", status="error", status_code=503, error_kind="server")
        with pytest.raises(LedgerError, match="already finalised"):
            ledger.finish_call("c1", status="ok")
        assert ledger.session_usage("session").attempts == 1


def test_concurrent_writes_and_idempotent_initialisation(tmp_path: Path) -> None:
    path = tmp_path / "usage.sqlite3"
    with Ledger(path) as first, Ledger(path) as second:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: (first if i % 2 else second).record_call(call(i)), range(40)))
        assert first.session_usage("session").attempts == 40
        assert second.session_usage("session").known_counted_tokens == 40 * 90


def test_local_attempts_do_not_consume_real_totals(tmp_path: Path) -> None:
    with Ledger(tmp_path / "usage.sqlite3") as ledger:
        ledger.record_call(call(1, real=False))
        assert ledger.requests_used_today("FAKE-model") == 0
        assert ledger.session_usage("session", real=False).attempts == 1
    with pytest.raises(ValueError, match="subset"):
        call(2, tokens_cached=101)
