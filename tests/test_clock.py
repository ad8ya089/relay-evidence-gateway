from datetime import UTC, datetime, timedelta

import pytest

from relay.obs.clock import FakeClock, SystemClock


def test_deterministic_elapsed_time_independent_of_wall_clock() -> None:
    now = datetime(2026, 11, 12, tzinfo=UTC)
    for _ in range(2):
        clock = FakeClock(utc=now, monotonic_ns=10)
        start = clock.monotonic_ns()
        clock.advance_ms(12.5)
        clock.set_utc(now - timedelta(days=1))
        assert (clock.monotonic_ns() - start) / 1_000_000 == 12.5
        assert clock.utc_now() == now - timedelta(days=1)
        assert clock.monotonic_ns() == clock.monotonic_ns()


def test_production_clock_and_timezone_normalisation() -> None:
    clock = SystemClock()
    assert clock.monotonic_ns() > 0
    assert clock.utc_now().utcoffset() == timedelta(0)
    fake = FakeClock(utc=datetime.fromisoformat("2026-11-12T05:30:00+05:30"))
    assert fake.utc_now() == datetime(2026, 11, 12, tzinfo=UTC)


def test_fake_clock_rejects_invalid_time() -> None:
    with pytest.raises(ValueError):
        FakeClock(utc=datetime(2026, 11, 12))
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))
    with pytest.raises(ValueError):
        clock.advance_ms(-1)
