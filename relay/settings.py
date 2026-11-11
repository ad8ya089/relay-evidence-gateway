"""Typed configuration, read once; importing this module performs no model/network work."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PaperId = Literal["transformer", "moshi", "dynamic_chunking"]
Setting = Literal["A", "B", "C", "D"]


class PaperSpec(BaseModel):
    paper_id: PaperId
    title: str
    arxiv_id: str
    arxiv_version: int = Field(gt=0)
    url: str


class ChunkSetting(BaseModel):
    target_tokens: int = Field(gt=0)
    overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    candidates: int = Field(gt=0)
    method: Literal["bm25", "hybrid"]
    rerank: bool

    @model_validator(mode="after")
    def valid_overlap(self) -> "ChunkSetting":
        if self.overlap >= self.target_tokens:
            raise ValueError("overlap must be smaller than target_tokens")
        return self


class RetrievalSettings(BaseModel):
    tokenizer: str
    dense_model: str
    rerank_model: str
    model_threads: int = Field(gt=0)
    model_batch_size: int = Field(gt=0)
    paper_hint_boost: float = Field(gt=1)
    rrf_k: int = Field(gt=0)
    dense_min_score: float = Field(ge=-1, le=1)
    timeout_ms: int = Field(gt=0)
    download_timeout_s: float = Field(gt=0)
    heading_font_ratio: float
    column_slack_ratio: float
    min_column_blocks: int
    max_heading_chars: int
    short_block_tokens: int
    chunk_tolerance: float
    settings: dict[Setting, ChunkSetting]
    papers: list[PaperSpec]


@lru_cache(maxsize=1)
def retrieval_settings() -> RetrievalSettings:
    with Path("config/retrieval.yaml").open(encoding="utf-8") as handle:
        return RetrievalSettings.model_validate(yaml.safe_load(handle))


class SafetySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELAY_", extra="ignore")
    provider: Literal["standin", "groq"] = "standin"
    allow_real: Literal["0", "1"] = "0"
    real_token_cap: int = Field(default=60000, gt=0)
    real_request_cap: int = Field(default=150, gt=0)
