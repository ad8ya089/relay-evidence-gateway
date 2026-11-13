"""Deterministic evidence checks; model text never supplies citation metadata."""

import re

from pydantic import ValidationError

from relay.app.models import ClaimCheck, Draft, ValidationResult
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.app.text import normalize
from relay.retrieval.models import RetrievalHit
from relay.settings import retrieval_settings


def metadata_errors(hit: RetrievalHit) -> list[str]:
    try:
        hit = RetrievalHit.model_validate(hit.model_dump())
    except ValidationError:
        return ["invalid_source_metadata"]
    spec = next(p for p in retrieval_settings().papers if p.paper_id == hit.paper_id)
    errors = []
    if (hit.arxiv_id, hit.arxiv_version, hit.title, hit.url) != (
        spec.arxiv_id,
        spec.arxiv_version,
        spec.title,
        spec.url,
    ):
        errors.append("pinned_source_mismatch")
    if not hit.section or not hit.section.strip():
        errors.append("missing_section")
    if hit.page_end < hit.page_start:
        errors.append("invalid_page_range")
    if not re.fullmatch(
        rf"{re.escape(hit.paper_id)}-v{hit.arxiv_version}-p{hit.page_start}-c\d+", hit.chunk_id
    ):
        errors.append("chunk_metadata_mismatch")
    return errors


def validate_evidence(
    draft: Draft, supplied: list[RetrievalHit], config: PipelineSettings | None = None
) -> ValidationResult:
    config = config or pipeline_settings()
    if draft.status != "answer":
        return ValidationResult(status="fail_final", failure_reason="not_an_answer")
    hits = {h.chunk_id: h for h in supplied}
    if len(hits) != len(supplied):
        return ValidationResult(status="fail_final", failure_reason="duplicate_supplied_ids")
    outcomes = []
    irreparable = False
    for i, claim in enumerate(draft.claims):
        reasons = []
        for evidence in claim.evidence:
            hit = hits.get(evidence.chunk_id)
            if hit is None:
                reasons.append("unknown_chunk_id")
                irreparable = True
                continue
            provenance = metadata_errors(hit)
            reasons.extend(provenance)
            irreparable |= bool(provenance)
            quote = normalize(evidence.quote)
            if not quote or quote not in normalize(hit.text):
                reasons.append("quote_not_substring")
            if len(evidence.quote.split()) > config.quote_max_words:
                reasons.append("quote_too_long")
        if re.search(r"\[[^\]]*(?:§|\bv\d+\b|\bpp?\.\s*\d)", claim.text):
            reasons.append("model_authored_citation")
        outcomes.append(
            ClaimCheck(
                i=i,
                passed=not reasons,
                checks={
                    "membership": "unknown_chunk_id" not in reasons,
                    "provenance": not any(
                        r in reasons
                        for r in (
                            "invalid_source_metadata",
                            "pinned_source_mismatch",
                            "missing_section",
                            "invalid_page_range",
                            "chunk_metadata_mismatch",
                        )
                    ),
                    "quote": not any(
                        r in reasons for r in ("quote_not_substring", "quote_too_long")
                    ),
                    "model_citations_absent": "model_authored_citation" not in reasons,
                },
                reasons=list(dict.fromkeys(reasons)),
            )
        )
    passed = all(c.passed for c in outcomes)
    return ValidationResult(
        status="pass" if passed else "fail_final" if irreparable else "fail_repairable",
        claims=outcomes,
        deterministic_passed=passed,
        repairable=not passed and not irreparable,
        failure_reason=None
        if passed
        else outcomes[next(i for i, c in enumerate(outcomes) if not c.passed)].reasons[0],
    )
