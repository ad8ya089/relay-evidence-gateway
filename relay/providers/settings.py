"""Typed provider transport and explicitly evidenced classifier configuration."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ErrorRule(BaseModel):
    kind: Literal["rate_limit_minute", "rate_limit_day"]
    field_path: str
    pattern: str
    source: Literal["observed", "fixture"]

    @model_validator(mode="after")
    def valid_pattern(self) -> "ErrorRule":
        re.compile(self.pattern)
        return self


class ProviderSettings(BaseModel):
    connect_timeout_s: float = Field(gt=0)
    read_timeout_s: float = Field(gt=0)
    write_timeout_s: float = Field(gt=0)
    pool_timeout_s: float = Field(gt=0)
    allowed_real_hosts: list[str]
    error_rules: list[ErrorRule]


@lru_cache(maxsize=1)
def provider_settings() -> ProviderSettings:
    return ProviderSettings.model_validate(
        yaml.safe_load(Path("config/provider.yaml").read_text(encoding="utf-8"))
    )
