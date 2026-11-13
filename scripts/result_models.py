"""Assignment CSV contract and strict logical-row validation."""

import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from loadgen.settings import Model

REQUIRED = (
    "request_id scenario source start_time first_token_ms total_ms status retries "
    "tokens_in tokens_out retrieval_setting answer_grade"
).split()
EXTRA = (
    "run_id queue_wait_ms retrieval_ms provider_ttft_ms provider_ms retry_wait_ms validation_ms "
    "provider_calls tokens_cached failure_reason http_status_last queue_len_at_arrival "
    "inflight_at_arrival mode cache_hit model citation_valid retrieval_hit deadline_ms"
).split()
AUDIT = (
    "usage_complete counted_tokens known_tokens_in known_tokens_out known_tokens_cached "
    "known_counted_tokens scheduled_arrival_monotonic actual_send_monotonic completion_monotonic "
    "generator_lag_ms transport_wait_ms scheduled_arrival_utc question_id answerable record_kind "
    "fixture_only terminal_complete client_status accounting_diagnostics queue_high_water "
    "peak_duplicate_provider_operations free_allocation server_cleanup_observed arrival_kind "
    "paper_id answer_disposition"
).split()
COLUMNS = REQUIRED + EXTRA + AUDIT


def timestamp(value: str) -> str:
    if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}(?:Z|\+00:00)", value) is None:
        raise ValueError("Timestamp must be UTC ISO-8601 with milliseconds")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Timestamp must be UTC")
    return value


class ResultRow(Model):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    request_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    source: Literal["real", "stand-in"]
    start_time: str
    first_token_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int | None = Field(default=None, ge=0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    retrieval_setting: Literal["A", "B", "C"]
    answer_grade: Literal["", "correct", "wrong", "refused_correctly", "refused_wrongly"] = ""
    run_id: str = Field(min_length=1)
    queue_wait_ms: float | None = Field(default=None, ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    provider_ttft_ms: float | None = Field(default=None, ge=0)
    provider_ms: float | None = Field(default=None, ge=0)
    retry_wait_ms: float | None = Field(default=None, ge=0)
    validation_ms: float | None = Field(default=None, ge=0)
    provider_calls: int | None = Field(default=None, ge=0)
    tokens_cached: int | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    http_status_last: int | None = None
    queue_len_at_arrival: int | None = Field(default=None, ge=0)
    inflight_at_arrival: int | None = Field(default=None, ge=0)
    mode: Literal["normal", "pressure", "exhausted"] | None = None
    cache_hit: bool | None = None
    model: str | None = None
    citation_valid: bool | None = None
    retrieval_hit: bool | None = None
    deadline_ms: int | None = Field(default=None, gt=0)
    usage_complete: bool | None = None
    counted_tokens: int | None = Field(default=None, ge=0)
    known_tokens_in: int | None = Field(default=None, ge=0)
    known_tokens_out: int | None = Field(default=None, ge=0)
    known_tokens_cached: int | None = Field(default=None, ge=0)
    known_counted_tokens: int | None = Field(default=None, ge=0)
    scheduled_arrival_monotonic: int | None = Field(default=None, ge=0)
    actual_send_monotonic: int | None = Field(default=None, ge=0)
    completion_monotonic: int | None = Field(default=None, ge=0)
    generator_lag_ms: float | None = Field(default=None, ge=0)
    transport_wait_ms: float | None = Field(default=None, ge=0)
    scheduled_arrival_utc: str | None = None
    question_id: str | None = None
    answerable: bool | None = None
    record_kind: Literal["load", "eval"] = "load"
    fixture_only: bool = False
    terminal_complete: bool | None = None
    client_status: str | None = None
    accounting_diagnostics: list[str] = Field(default_factory=list)
    queue_high_water: int | None = Field(default=None, ge=0)
    peak_duplicate_provider_operations: int | None = Field(default=None, ge=0)
    free_allocation: bool | None = None
    server_cleanup_observed: bool | None = None
    arrival_kind: Literal["constant", "step", "burst", "ramp", "one_shot"] | None = None
    paper_id: str | None = None
    answer_disposition: Literal["answered", "refused", "failed"] | None = None

    _timestamp = field_validator("start_time")(timestamp)

    @model_validator(mode="after")
    def consistent(self) -> "ResultRow":
        if (
            any(not value.strip() for value in (self.request_id, self.run_id, self.scenario))
            or self.client_status
            and self.client_status != self.status
        ):
            raise ValueError("Missing identifier or contradictory client outcome")
        if self.scheduled_arrival_utc is not None:
            timestamp(self.scheduled_arrival_utc)
        if (
            self.actual_send_monotonic is not None
            and self.completion_monotonic is not None
            and self.completion_monotonic < self.actual_send_monotonic
        ):
            raise ValueError("Completion precedes dispatch")
        if self.record_kind == "load":
            if self.answer_grade:
                raise ValueError("Load-test grades must remain empty")
            if self.status == "ok" and (
                self.terminal_complete is not True
                or self.http_status_last != 200
                or self.first_token_ms is None
            ):
                raise ValueError(
                    "Successful load rows require complete client SSE and first answer"
                )
        if self.first_token_ms is not None and self.first_token_ms > self.total_ms:
            raise ValueError("First answer exceeds total duration")
        if (
            self.tokens_in is not None
            and self.tokens_cached is not None
            and self.tokens_cached > self.tokens_in
        ):
            raise ValueError("Cached input exceeds input")
        if self.usage_complete and any(
            v is None for v in (self.tokens_in, self.tokens_out, self.tokens_cached)
        ):
            raise ValueError("Complete usage cannot have unknown token fields")
        return self
