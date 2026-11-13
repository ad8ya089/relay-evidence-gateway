"""Application contracts reuse source-owned retrieval and provider usage models."""

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from relay.app.settings import pipeline_settings
from relay.providers.base import Usage
from relay.retrieval.models import RetrievalHit, RetrievalResult
from relay.settings import PaperId, Setting

__all__ = ["RetrievalHit", "RetrievalResult"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Question(StrictModel):
    text: str
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    setting: Setting | None = None
    deadline_ms: int | None = Field(default=None, gt=0)
    paper_scope: PaperId | None = None

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        if not value.strip() or len(value) > pipeline_settings().max_question_chars:
            raise ValueError("Question is blank or exceeds configured length")
        return value.strip()


class Interpretation(StrictModel):
    paper_hint: PaperId | None = None
    explicit_paper: PaperId | None = None
    mentioned_papers: list[PaperId] = Field(default_factory=list)
    qtype: Literal["explain", "compare", "number", "limitation", "absent"]
    numeric: bool
    looks_out_of_scope: bool
    keywords: list[str]
    diagnostics: list[str] = Field(default_factory=list)


class Evidence(StrictModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class DraftClaim(StrictModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def bounded_claim(cls, value: str) -> str:
        if not value.strip() or len(value) > pipeline_settings().max_claim_chars:
            raise ValueError("Claim is blank or exceeds configured length")
        return value


class Draft(StrictModel):
    status: Literal["answer", "insufficient_evidence"]
    claims: list[DraftClaim]
    note: str = ""

    @model_validator(mode="after")
    def consistent_draft(self) -> "Draft":
        config = pipeline_settings()
        if (self.status == "answer") != bool(self.claims):
            raise ValueError("Answers require claims; refusals must have none")
        if len(self.claims) > config.max_claims or len(self.note.split()) > config.note_max_words:
            raise ValueError("Draft exceeds configured bounds")
        return self


class VerifyVerdict(StrictModel):
    i: int = Field(ge=0, strict=True)
    v: Literal["supported", "unsupported", "overstated"]
    why: str

    @field_validator("why")
    @classmethod
    def bounded_reason(cls, value: str) -> str:
        if len(value.split()) > pipeline_settings().verdict_reason_max_words:
            raise ValueError("Verifier reason exceeds configured word bound")
        return value


class Verification(StrictModel):
    verdicts: list[VerifyVerdict] = Field(min_length=1)


class ClaimCheck(StrictModel):
    i: int
    passed: bool
    checks: dict[str, bool]
    reasons: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    status: Literal["pass", "fail_repairable", "fail_final", "verify_invalid"]
    claims: list[ClaimCheck] = Field(default_factory=list)
    deterministic_passed: bool = False
    verifier_passed: bool = False
    verdicts: list[VerifyVerdict] = Field(default_factory=list)
    verifier_diagnostics: list[str] = Field(default_factory=list)
    repairable: bool = False
    failure_reason: str | None = None


class Citation(StrictModel):
    chunk_id: str
    paper_id: PaperId
    arxiv_id: str
    arxiv_version: int = Field(gt=0)
    title: str
    section: str
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    display: str


class State(StrEnum):
    ARRIVED = "ARRIVED"
    INTERPRETED = "INTERPRETED"
    RETRIEVED = "RETRIEVED"
    GATE = "GATE"
    BROADEN = "BROADEN"
    DRAFTING = "DRAFTING"
    VALIDATING = "VALIDATING"
    REPAIRING = "REPAIRING"
    RELEASED = "RELEASED"
    REFUSED_LOCAL = "REFUSED_LOCAL"
    REFUSED_MODEL = "REFUSED_MODEL"
    ABSTAINED = "ABSTAINED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"


class CallAttempt(StrictModel):
    call_id: str
    kind: Literal["draft", "verify", "repair"]
    dispatched: bool
    usage: Usage
    usage_complete: bool
    total_ms: float = Field(ge=0)
    provider_ttft_ms: float | None = None
    status_code: int | None = None
    failure_kind: str | None = None
    ledger_reconciled: bool = False
    diagnostics: list[str] = Field(default_factory=list)


class TokenTotals(StrictModel):
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    tokens_cached: int | None = 0
    counted_tokens: int | None = 0
    known_tokens_in: int = 0
    known_tokens_out: int = 0
    known_tokens_cached: int = 0
    known_counted_tokens: int = 0
    unknown_usage_calls: int = 0
    complete: bool = True


class Outcome(StrictModel):
    request_id: str
    run_id: str
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    state: State
    status: Literal["ok", "error", "timeout", "rate_limited"]
    failure_reason: str | None = None
    validated_refusal: bool = False
    refusal_reason: str | None = None
    releasable: bool = False
    draft_calls: int = 0
    verifier_calls: int = 0
    repair_calls: int = 0
    retries: Literal[0] = 0
    usage: TokenTotals = Field(default_factory=TokenTotals)
    calls: list[CallAttempt] = Field(default_factory=list)
    stage_ms: dict[str, float] = Field(default_factory=dict)
    provider_ms: float = 0
    provider_ttft_ms: float | None = None
    total_ms: float = 0
    first_user_token_at: int | None = None
    first_token_ms: float | None = None
    release_ready_at_ns: int | None = None
    arrival_ns: int = Field(ge=0, exclude=True)
    deadline_ns: int = Field(ge=0, exclude=True)
    retrieval_hit: bool | None = None  # Human gold-key hit is not computed by P4.
    setting: Setting
    supplied_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_diagnostics: list[str] = Field(default_factory=list)
    validation: ValidationResult | None = None
    states: list[State] = Field(default_factory=list)
    _delivery_seal: str | None = PrivateAttr(default=None)
