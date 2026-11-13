"""P4 configuration only; retrieval/model settings remain owned by P1/P2/P3."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from relay.settings import Setting


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    status: Literal["assumed", "chosen"]
    default_setting: Setting
    deadline_ms: int = Field(gt=0)
    max_question_chars: int = Field(gt=0)
    max_claim_chars: int = Field(gt=0)
    max_claims: int = Field(gt=0)
    quote_max_words: int = Field(gt=0, le=40)
    note_max_words: int = Field(gt=0, le=20)
    verdict_reason_max_words: int = Field(gt=0, le=15)
    broaden_max: int = Field(ge=0, le=2)
    repair_max: int = Field(ge=0, le=1)
    repair_min_remaining_ms: int = Field(ge=0)
    draft_max_completion_tokens: int = Field(gt=0)
    verify_max_completion_tokens: int = Field(gt=0)
    draft_timeout_ms: int = Field(gt=0)
    verify_timeout_ms: int = Field(gt=0)
    draft_stream: bool
    verify_stream: bool
    reasoning_effort: str
    tool_timeout_ms: int = Field(gt=0)
    tool_retry_max: int = Field(ge=0, le=1)
    tool_retry_min_remaining_ms: int = Field(ge=0)
    tool_fault_prob: float = Field(ge=0, le=1)
    seed: int
    gate_min_score: float
    gate_min_keyword_coverage: float = Field(ge=0, le=1)
    gate_min_matching_terms: int = Field(ge=1)
    broadening_term_max: int = Field(gt=0)
    numeric_context_chars: int = Field(gt=0)
    delivery_chunk_chars: int = Field(gt=0)
    stopwords: list[str]
    section_terms: dict[str, list[str]]


class ToolEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    tool_fault_prob: float | None = Field(default=None, ge=0, le=1)


@lru_cache(maxsize=1)
def pipeline_settings() -> PipelineSettings:
    return PipelineSettings.model_validate(
        yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    )
