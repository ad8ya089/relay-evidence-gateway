"""Minimal injectable duration clock; event logging is a P2 task."""

import time
from typing import Protocol


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()
