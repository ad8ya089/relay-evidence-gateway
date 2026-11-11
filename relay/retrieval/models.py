"""Source-owned metadata and the architecture's retrieval contract."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from relay.settings import PaperId, PaperSpec, Setting

ChunkKind = Literal["text", "table", "caption", "references", "appendix"]


class Chunk(PaperSpec):
    chunk_id: str
    section: str | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    kind: ChunkKind
    text: str = Field(min_length=1)
    n_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered_pages(self) -> "Chunk":
        if self.page_end < self.page_start:
            raise ValueError("page_end precedes page_start")
        return self


class RetrievalHit(BaseModel):
    chunk_id: str
    paper_id: PaperId
    arxiv_id: str
    arxiv_version: int
    title: str
    url: str
    section: str | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str
    score: float
    rank: int
    # Additive diagnostics preserve the section 4.1 public fields.
    kind: ChunkKind
    n_tokens: int
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    hint_multiplier: float = 1.0
    neighbour: bool = False


class RetrievalResult(BaseModel):
    status: Literal["ok", "empty", "error", "timeout"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    setting: Setting
    query_used: str
    took_ms: float = 0
    attempt: int = 1
    rerank_method: Literal["none", "cross_encoder"] = "none"
    diagnostics: list[str] = Field(default_factory=list)
    method: Literal["bm25", "hybrid"]


class DocumentManifest(PaperSpec):
    sha256: str
    page_count: int = Field(gt=0)
    fetched_at: str


class CorpusManifest(BaseModel):
    documents: list[DocumentManifest]
    corpus_hash: str
