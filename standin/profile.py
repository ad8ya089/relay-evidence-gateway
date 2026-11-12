"""Typed simulation parameters with explicit provenance, never implicit calibration."""

import math
from functools import lru_cache
from pathlib import Path
from typing import Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class Parameter(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: T
    source: Literal["fitted", "chosen", "assumed", "observed"]
    note: str = Field(min_length=1)
    observed_at: str | None = None

    @model_validator(mode="after")
    def observed_provenance(self) -> "Parameter[T]":
        if self.source == "observed" and not self.observed_at:
            raise ValueError("Observed parameters require an observation date")
        return self


class Distribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family: Literal["constant", "lognormal", "empirical"]
    parameters: Parameter[list[float]]

    @model_validator(mode="after")
    def valid_values(self) -> "Distribution":
        values = self.parameters.value
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError("Distribution values must be finite and nonempty")
        if self.family == "constant" and len(values) != 1:
            raise ValueError("Constant distributions require one value")
        if self.family == "lognormal":
            if len(values) != 2 or values[1] < 0:
                raise ValueError("Lognormal parameters are [mu, nonnegative sigma]")
        elif any(v <= 0 for v in values):
            raise ValueError("Samples must be positive")
        return self


class Timing(BaseModel):
    ttft_ms: Distribution
    decode_tps: Distribution
    completion_tokens: Distribution


class Limits(BaseModel):
    rpm: Parameter[int | None]
    tpm: Parameter[int | None]
    rpd: Parameter[int | None]
    tpd: Parameter[int | None]

    @model_validator(mode="after")
    def positive_limits(self) -> "Limits":
        for field in type(self).model_fields:
            value = getattr(self, field).value
            if value is not None and value <= 0:
                raise ValueError("Quota limits must be positive or unknown")
        return self

    def numbers(self) -> dict[str, int]:
        missing = [name for name in type(self).model_fields if getattr(self, name).value is None]
        if missing:
            raise ValueError("Account Limits-page inputs required: " + ", ".join(missing))
        return {name: getattr(self, name).value for name in type(self).model_fields}


class FaultSettings(BaseModel):
    server_error_prob: Parameter[float]
    timeout_prob: Parameter[float]
    disconnect_prob: Parameter[float]
    slow_success_prob: Parameter[float]
    verify_fail_prob: Parameter[float]
    slow_factor: Parameter[float]
    server_status: Parameter[Literal[500, 503]]
    timeout_s: Parameter[float]
    disconnect_fraction: Parameter[float]

    @model_validator(mode="after")
    def valid_faults(self) -> "FaultSettings":
        for name in type(self).model_fields:
            value = getattr(self, name).value
            if not math.isfinite(value):
                raise ValueError("Fault parameters must be finite")
            if name.endswith("_prob") and not 0 <= value <= 1:
                raise ValueError("Fault probabilities must be in [0,1]")
        if self.slow_factor.value < 1 or self.timeout_s.value <= 0:
            raise ValueError("Slow factor >=1 and timeout >0 required")
        if not 0 < self.disconnect_fraction.value < 1:
            raise ValueError("Disconnect fraction must be inside (0,1)")
        return self


class CacheSettings(BaseModel):
    enabled: Parameter[bool]
    ttl_s: Parameter[float]
    minimum_tokens: Parameter[int]
    max_entries: Parameter[int]

    @model_validator(mode="after")
    def valid_cache(self) -> "CacheSettings":
        if not math.isfinite(self.ttl_s.value) or self.ttl_s.value <= 0:
            raise ValueError("Cache TTL must be positive and finite")
        if self.minimum_tokens.value < 0 or self.max_entries.value <= 0:
            raise ValueError("Invalid cache threshold/capacity")
        return self


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    description: str
    model: str
    models: dict[str, Limits] = Field(min_length=1)
    seed: Parameter[int]
    capacity: Parameter[int]
    alpha: Parameter[float]
    chunks: Parameter[int]
    reasoning_fraction: Parameter[float]
    window_s: Parameter[Literal[60]]
    disconnect_poll_s: Parameter[float]
    diagnostic_history: Parameter[int]
    daily_reset: Literal["UTC midnight"]
    reset_format: Literal["simulated decimal seconds with s suffix"]
    timing: dict[Literal["draft", "verify", "generic"], Timing]
    faults: FaultSettings
    cache: CacheSettings
    scale: Parameter[int]

    @model_validator(mode="after")
    def valid_profile(self) -> "Profile":
        if self.model not in self.models or set(self.timing) != {"draft", "verify", "generic"}:
            raise ValueError("Default model and all call-kind timing profiles required")
        if (
            min(
                self.capacity.value,
                self.chunks.value,
                self.diagnostic_history.value,
                self.scale.value,
            )
            <= 0
        ):
            raise ValueError("Capacity, chunks, history and scale must be positive")
        if not math.isfinite(self.alpha.value) or self.alpha.value < 0:
            raise ValueError("Alpha must be finite and nonnegative")
        if not 0 <= self.reasoning_fraction.value <= 1:
            raise ValueError("Reasoning fraction must be in [0,1]")
        if not math.isfinite(self.disconnect_poll_s.value) or self.disconnect_poll_s.value <= 0:
            raise ValueError("Polling interval must be positive and finite")
        return self

    def runnable(self) -> "Profile":
        for limits in self.models.values():
            limits.numbers()
        return self


@lru_cache(maxsize=1)
def profiles() -> dict[str, Profile]:
    raw = yaml.safe_load(Path("config/standin_profiles.yaml").read_text(encoding="utf-8"))
    return {
        name: Profile.model_validate(data | {"name": name})
        for name, data in raw["profiles"].items()
    }


def load_profile(name: str = "development") -> Profile:
    catalog = profiles()
    if name not in catalog:
        raise ValueError("Unknown stand-in profile")
    profile = catalog[name]
    if name == "scaled":
        base = catalog["free_exact"].runnable()
        models = {}
        for model, limits in base.models.items():
            models[model] = Limits.model_validate(
                {
                    field: Parameter[int](
                        value=value * profile.scale.value,
                        source="chosen",
                        note=f"free_exact {field} multiplied by K={profile.scale.value}",
                    )
                    for field, value in limits.numbers().items()
                }
            )
        profile = profile.model_copy(update={"models": models, "model": base.model})
    return profile.runnable()
