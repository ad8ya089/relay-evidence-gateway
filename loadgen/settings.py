"""Typed experiment controls; no provider facts or credential discovery."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CapacitySettings(Model):
    rates_per_minute: list[float] = Field(min_length=2)
    duration_s: float = Field(gt=0)
    minimum_success: float = Field(gt=0, le=1)
    minimum_samples: int = Field(gt=0)
    tiny_rates_per_minute: list[float] = Field(min_length=2)
    tiny_duration_s: float = Field(gt=0)
    faults: dict[str, float]

    @model_validator(mode="after")
    def ordered(self) -> "CapacitySettings":
        for rates in (self.rates_per_minute, self.tiny_rates_per_minute):
            if min(rates) <= 0 or any(a >= b for a, b in zip(rates, rates[1:], strict=False)):
                raise ValueError("Capacity rates must strictly increase")
        return self


class LoadSettings(Model):
    status: Literal["assumed", "chosen"]
    max_requests: int = Field(gt=0)
    max_connections: int = Field(gt=0)
    max_keepalive_connections: int = Field(ge=0)
    connect_timeout_s: float = Field(gt=0)
    pool_timeout_s: float = Field(gt=0)
    max_sse_frame_bytes: int = Field(gt=0)
    lag_warning_ms: float = Field(gt=0)
    health_poll_s: float = Field(gt=0)
    startup_timeout_s: float = Field(gt=0)
    shutdown_timeout_s: float = Field(gt=0)
    admin_timeout_s: float = Field(gt=0)
    quiescence_timeout_s: float = Field(gt=0)
    min_percentile_samples: int = Field(gt=0)
    expected_questions: int = Field(gt=0)
    expected_answerable: int = Field(gt=0)
    zipf_exponent: float = Field(gt=0)
    capacity: CapacitySettings

    @model_validator(mode="after")
    def pool_size(self) -> "LoadSettings":
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError("Keepalive count exceeds connection pool")
        return self


def load_settings(path: Path = Path("config/loadgen.yaml")) -> LoadSettings:
    return LoadSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
