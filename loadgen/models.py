"""Raw measurement contracts; nullable fields distinguish unknown from measured zero."""

from typing import Literal

from pydantic import ConfigDict, Field

from loadgen.settings import Model


class RunContext(Model):
    run_id: str
    scenario: str
    source: Literal["real", "stand-in"] = "stand-in"
    model: str
    deadline_ms: int = Field(gt=0)
    cache: bool
    fixture_only: bool = False


class ClientRecord(Model):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    scenario: str
    source: Literal["real", "stand-in"]
    question_id: str
    seed: int
    sampling: Literal["uniform", "zipf"]
    sample_draw: float = Field(ge=0, lt=1)
    fixture_only: bool
    scheduled_arrival_monotonic: int = Field(ge=0)
    task_dispatch_monotonic: int = Field(ge=0)
    actual_send_monotonic: int = Field(ge=0)
    first_answer_byte_monotonic: int | None = Field(default=None, ge=0)
    completion_monotonic: int = Field(ge=0)
    scheduled_arrival_utc: str
    actual_send_utc: str
    dispatch_timestamp_source: str
    generator_lag_ms: float = Field(ge=0)
    transport_wait_ms: float = Field(ge=0)
    first_token_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)
    status: Literal["ok", "rate_limited", "timeout", "error"]
    failure_reason: str | None = None
    http_status_last: int | None = None
    retries: int | None = Field(default=None, ge=0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    tokens_cached: int | None = Field(default=None, ge=0)
    counted_tokens: int | None = Field(default=None, ge=0)
    provider_calls: int | None = Field(default=None, ge=0)
    queue_wait_ms: float | None = None
    retrieval_ms: float | None = None
    provider_ttft_ms: float | None = None
    provider_ms: float | None = None
    retry_wait_ms: float | None = None
    validation_ms: float | None = None
    queue_len_at_arrival: int | None = None
    inflight_at_arrival: int | None = None
    mode: str | None = None
    cache_hit: bool | None = None
    coalesced: bool | None = None
    model: str
    citation_valid: bool | None = None
    retrieval_hit: bool | None = None
    deadline_ms: int
    usage_complete: bool | None = None
    terminal_complete: bool = False
    server_cleanup_observed: bool | None = None
    retrieval_setting: Literal["A", "B", "C"]
