"""Injectable monotonic durations and UTC persistence timestamps."""

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...
    def utc_now(self) -> datetime: ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Reads never advance time; tests explicitly advance or change the UTC date."""

    def __init__(self, *, utc: datetime, monotonic_ns: int = 0) -> None:
        if utc.tzinfo is None or utc.utcoffset() is None:
            raise ValueError("FakeClock requires a timezone-aware timestamp")
        if monotonic_ns < 0:
            raise ValueError("monotonic_ns must be nonnegative")
        self._utc = utc.astimezone(UTC)
        self._ns = monotonic_ns
        self._lock = threading.Lock()

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._ns

    def utc_now(self) -> datetime:
        with self._lock:
            return self._utc

    def advance_ms(self, milliseconds: float) -> None:
        if milliseconds < 0:
            raise ValueError("a monotonic clock cannot go backwards")
        with self._lock:
            self._ns += int(milliseconds * 1_000_000)
            self._utc += timedelta(milliseconds=milliseconds)

    def set_utc(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("UTC timestamps must be timezone-aware")
        with self._lock:
            self._utc = timestamp.astimezone(UTC)
