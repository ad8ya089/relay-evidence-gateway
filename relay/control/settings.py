"""Validated P5 engineering policy; no account discovery."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from relay.providers.settings import ErrorRule


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Limits(PolicyModel):
    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    rpd: int = Field(gt=0)
    tpd: int = Field(gt=0)


class Estimate(PolicyModel):
    draft_input: int = Field(gt=0)
    verify_input: int = Field(gt=0)
    expected_service_ms: int = Field(gt=0)


class ModePolicy(PolicyModel):
    top_k: int | None = Field(ge=1)
    draft_max_completion_tokens: int = Field(gt=0)
    verify_max_completion_tokens: int = Field(gt=0)
    repair_max: int = Field(ge=0, le=1)


class RetrySettings(PolicyModel):
    policy: Literal["bounded", "naive_s7a"]
    max_per_request: int = Field(ge=0, le=1)
    budget_fraction: float = Field(ge=0, le=1)
    budget_capacity: int = Field(ge=1)
    base_backoff_ms: int = Field(ge=0)
    max_backoff_ms: int = Field(ge=0)
    safety_margin_ms: int = Field(ge=0)
    seed: int
    naive_timeout_ms: int = Field(gt=0)
    naive_retries: int = Field(ge=0)


class BreakerSettings(PolicyModel):
    consecutive_failures: int = Field(gt=0)
    failure_window_ms: int = Field(gt=0)
    open_ms: int = Field(gt=0)
    count_minute_429: bool


class CacheSettings(PolicyModel):
    enabled: bool
    ttl_s: float = Field(gt=0)
    max_entries: int = Field(gt=0)
    singleflight: bool
    max_waiters: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)


class SchedulerSettings(PolicyModel):
    priorities: dict[Literal["draft", "verify", "repair"], int]
    max_waiters: int = Field(gt=0)
    aging_ms: int = Field(gt=0)
    max_priority_burst: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_priorities(self) -> "SchedulerSettings":
        if set(self.priorities) != {"draft", "verify", "repair"} or not (
            self.priorities["verify"] < self.priorities["repair"] < self.priorities["draft"]
        ):
            raise ValueError("Priorities must order verify, repair, draft")
        return self


class ServerSettings(PolicyModel):
    host: Literal["127.0.0.1"]
    port: int = Field(ge=0, le=65535)
    disconnect_poll_ms: int = Field(gt=0)
    graceful_shutdown_s: int = Field(gt=0)
    request_id_max_chars: int = Field(gt=0)
    standin_base_url: str


class Policy(PolicyModel):
    provider: Literal["standin"]
    status: Literal["chosen", "assumed"]
    deadline_ms: int = Field(gt=0)
    max_queue_wait_ms: int = Field(ge=0)
    queue_capacity: int = Field(ge=0)
    max_active_requests: int = Field(gt=0)
    max_inflight_provider: int = Field(gt=0)
    max_inflight_real: int = Field(gt=0)
    window_s: Literal[60]
    daily_reset_known: bool
    standin_error_rules: list[ErrorRule]
    limits: dict[str, Limits] = Field(min_length=1)
    estimates: dict[Literal["A", "B", "C"], Estimate]
    normal: ModePolicy
    pressure: ModePolicy
    pressure_minute_util: float = Field(gt=0, le=1)
    pressure_day_util: float = Field(gt=0, le=1)
    safety_tokens: int = Field(ge=0)
    retry: RetrySettings
    breaker: BreakerSettings
    cache: CacheSettings
    scheduler: SchedulerSettings
    server: ServerSettings

    @model_validator(mode="after")
    def complete_policy(self) -> "Policy":
        if set(self.estimates) != {"A", "B", "C"}:
            raise ValueError("All online retrieval estimates are required")
        if self.pressure.top_k is None or self.pressure.top_k > 3:
            raise ValueError("Pressure evidence cap must be at most three")
        if (
            self.pressure.repair_max
            or (
                self.pressure.draft_max_completion_tokens >= self.normal.draft_max_completion_tokens
            )
            or self.pressure.verify_max_completion_tokens < self.normal.verify_max_completion_tokens
        ):
            raise ValueError("Pressure reduces draft cap, preserves verifier and disables repair")
        if self.retry.base_backoff_ms > self.retry.max_backoff_ms:
            raise ValueError("Backoff base exceeds cap")
        return self


def load_policy(path: Path = Path("config/policy.yaml")) -> Policy:
    return Policy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
