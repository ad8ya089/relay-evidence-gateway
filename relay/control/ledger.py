"""SQLite attempt accounting. Unknown usage is retained, never invented as zero."""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, model_validator

from relay.obs.clock import Clock, SystemClock


class LedgerError(RuntimeError):
    """Accounting could not be persisted or read safely."""


class DuplicateCallError(LedgerError):
    """A call ID already exists; a retry needs its own ID."""


class SessionCapExceeded(LedgerError):
    """A reservation would exceed a session cap, or prior usage is unbounded."""


class LedgerSettings(BaseModel):
    busy_timeout_ms: int = Field(gt=0)


@lru_cache(maxsize=1)
def ledger_settings() -> LedgerSettings:
    return LedgerSettings.model_validate(
        yaml.safe_load(Path("config/ledger.yaml").read_text(encoding="utf-8"))
    )


class CallRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    ts_utc: datetime | None = None
    kind: str
    model: str
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    tokens_cached: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    status: str
    real: bool
    session_id: str
    request_id: str
    status_code: int | None = None
    error_kind: str | None = None
    reserved_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent_usage(self) -> "CallRecord":
        if self.ts_utc is not None:
            if self.ts_utc.tzinfo is None or self.ts_utc.utcoffset() is None:
                raise ValueError("ledger timestamps must be timezone-aware")
            self.ts_utc = self.ts_utc.astimezone(UTC)
        if self.tokens_in is not None and self.tokens_cached is not None:
            if self.tokens_cached > self.tokens_in:
                raise ValueError("cached tokens must be a subset of input tokens")
        return self

    @property
    def counted_tokens(self) -> int | None:
        if self.tokens_in is None or self.tokens_out is None or self.tokens_cached is None:
            return None
        return self.tokens_in - self.tokens_cached + self.tokens_out


class UsageSummary(BaseModel):
    attempts: int
    known_counted_tokens: int
    reported_total_tokens: int
    unknown_usage_count: int
    unknown_total_count: int
    inflight_attempts: int
    guard_charged_tokens: int
    unreserved_unknown_count: int
    inconsistent_usage_count: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, ts_utc TEXT NOT NULL,
 kind TEXT NOT NULL, model TEXT NOT NULL,
 tokens_in INTEGER, tokens_out INTEGER, tokens_cached INTEGER,
 status TEXT NOT NULL, real INTEGER NOT NULL,
 session_id TEXT NOT NULL, request_id TEXT NOT NULL, status_code INTEGER,
 error_kind TEXT, total_tokens INTEGER, usage_known INTEGER NOT NULL,
 counted_tokens INTEGER, reserved_tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_model_date ON calls(model, real, ts_utc);
CREATE INDEX IF NOT EXISTS calls_session ON calls(session_id, real);
CREATE INDEX IF NOT EXISTS calls_request ON calls(request_id);
"""
SUMMARY = """
SELECT COUNT(*) AS attempts,
 COALESCE(SUM(counted_tokens), 0) AS known_counted_tokens,
 COALESCE(SUM(total_tokens), 0) AS reported_total_tokens,
 COALESCE(SUM(usage_known = 0), 0) AS unknown_usage_count,
 COALESCE(SUM(total_tokens IS NULL), 0) AS unknown_total_count,
 COALESCE(SUM(status = 'pending'), 0) AS inflight_attempts,
 COALESCE(SUM(COALESCE(counted_tokens, reserved_tokens)), 0) AS guard_charged_tokens,
 COALESCE(SUM(counted_tokens IS NULL AND reserved_tokens = 0), 0) AS unreserved_unknown_count,
 COALESCE(SUM(total_tokens IS NOT NULL AND tokens_in IS NOT NULL AND tokens_out IS NOT NULL
              AND total_tokens != tokens_in + tokens_out), 0) AS inconsistent_usage_count
FROM calls WHERE
"""


class Ledger:
    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        self.clock = clock if clock is not None else SystemClock()
        self._lock = threading.RLock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(
                path,
                timeout=ledger_settings().busy_timeout_ms / 1000,
                check_same_thread=False,
                isolation_level=None,
            )
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Cannot initialise ledger ({type(exc).__name__})") from None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _insert(self, record: CallRecord) -> None:
        values = record.model_dump()
        values["ts_utc"] = (record.ts_utc or self.clock.utc_now()).astimezone(UTC).isoformat()
        values["real"] = int(record.real)
        values["counted_tokens"] = record.counted_tokens
        values["usage_known"] = int(record.counted_tokens is not None)
        self._db.execute(
            """INSERT INTO calls (
            id, run_id, ts_utc, kind, model, tokens_in, tokens_out, tokens_cached,
            status, real, session_id, request_id, status_code, error_kind,
            total_tokens, usage_known, counted_tokens, reserved_tokens
            ) VALUES (:id, :run_id, :ts_utc, :kind, :model, :tokens_in, :tokens_out,
            :tokens_cached, :status, :real, :session_id, :request_id, :status_code,
            :error_kind, :total_tokens, :usage_known, :counted_tokens, :reserved_tokens)""",
            values,
        )

    def record_call(self, record: CallRecord) -> None:
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._insert(record)
                    self._db.commit()
                except (sqlite3.Error, ValueError):
                    self._db.rollback()
                    raise
        except sqlite3.IntegrityError:
            raise DuplicateCallError("Call ID already recorded; retries require a new ID") from None
        except sqlite3.Error as exc:
            raise LedgerError(f"Cannot record attempt ({type(exc).__name__})") from None

    def reserve_attempt(self, record: CallRecord, *, token_cap: int, request_cap: int) -> None:
        """Atomic across connections/processes: pending attempts already consume caps."""
        if not record.real or record.status != "pending" or record.reserved_tokens <= 0:
            raise ValueError("Real reservations require a pending attempt and positive token bound")
        if record.counted_tokens is not None:
            raise ValueError("A reservation cannot already claim observed usage")
        try:
            with self._lock:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    usage = self._summary("session_id = ? AND real = 1", (record.session_id,))
                    if usage.unreserved_unknown_count or usage.inconsistent_usage_count:
                        raise SessionCapExceeded("Prior unknown/unbounded usage requires review")
                    if usage.attempts + 1 > request_cap:
                        raise SessionCapExceeded("RELAY_REAL_REQUEST_CAP exhausted")
                    if usage.guard_charged_tokens + record.reserved_tokens > token_cap:
                        raise SessionCapExceeded("RELAY_REAL_TOKEN_CAP would be exceeded")
                    self._insert(record)
                    self._db.commit()
                except (sqlite3.Error, LedgerError, ValueError):
                    self._db.rollback()
                    raise
        except sqlite3.IntegrityError:
            raise DuplicateCallError("Call ID already recorded") from None
        except sqlite3.Error as exc:
            raise LedgerError(f"Cannot reserve attempt ({type(exc).__name__})") from None

    def _summary(self, where: str, parameters: tuple[object, ...]) -> UsageSummary:
        row = self._db.execute(SUMMARY + where, parameters).fetchone()
        return UsageSummary.model_validate(dict(row))

    def session_usage(self, session_id: str, *, real: bool = True) -> UsageSummary:
        try:
            with self._lock:
                return self._summary("session_id = ? AND real = ?", (session_id, int(real)))
        except sqlite3.Error as exc:
            raise LedgerError(f"Cannot read session usage ({type(exc).__name__})") from None

    def daily_usage(self, model: str, *, real: bool = True) -> UsageSummary:
        start = (
            self.clock.utc_now().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        end = start + timedelta(days=1)
        try:
            with self._lock:
                return self._summary(
                    "model = ? AND real = ? AND ts_utc >= ? AND ts_utc < ?",
                    (model, int(real), start.isoformat(), end.isoformat()),
                )
        except sqlite3.Error as exc:
            raise LedgerError(f"Cannot read daily usage ({type(exc).__name__})") from None

    def tokens_used_today(self, model: str) -> int:
        """Known counted tokens only; daily_usage also exposes the unknown count."""
        return self.daily_usage(model).known_counted_tokens

    def requests_used_today(self, model: str) -> int:
        """Local outbound attempts, including 429s; not provider quota consumption."""
        return self.daily_usage(model).attempts

    def get_call(self, call_id: str) -> CallRecord:
        try:
            with self._lock:
                row = self._db.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
            if row is None:
                raise LedgerError("Unknown call ID")
            return CallRecord.model_validate(dict(row))
        except sqlite3.Error as exc:
            raise LedgerError(f"Cannot read attempt ({type(exc).__name__})") from None

    def finish_call(
        self,
        call_id: str,
        *,
        status: str,
        status_code: int | None = None,
        error_kind: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        tokens_cached: int | None = None,
        total_tokens: int | None = None,
    ) -> CallRecord:
        with self._lock:
            previous = self.get_call(call_id)
            values = previous.model_dump() | {
                "status": status,
                "status_code": status_code,
                "error_kind": error_kind,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_cached": tokens_cached,
                "total_tokens": total_tokens,
            }
            record = CallRecord.model_validate(values)
            values["counted_tokens"] = record.counted_tokens
            values["usage_known"] = int(record.counted_tokens is not None)
            try:
                cursor = self._db.execute(
                    """UPDATE calls SET status=:status, status_code=:status_code,
                    error_kind=:error_kind, tokens_in=:tokens_in, tokens_out=:tokens_out,
                    tokens_cached=:tokens_cached, total_tokens=:total_tokens,
                    counted_tokens=:counted_tokens, usage_known=:usage_known
                    WHERE id=:id AND status='pending'""",
                    values,
                )
                if cursor.rowcount != 1:
                    raise LedgerError("Attempt is already finalised")
            except sqlite3.Error as exc:
                raise LedgerError(f"Cannot finalise attempt ({type(exc).__name__})") from None
            return record
