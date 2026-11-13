"""Render only verified claims, with citations derived exclusively from source hits."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from relay.app.models import Citation, Draft, ValidationResult
from relay.app.validation import metadata_errors, validate_evidence
from relay.retrieval.models import RetrievalHit


class RenderError(ValueError):
    """No validated answer can be rendered from these inputs."""


class RenderedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    citations: list[Citation]
    claim_citations: dict[int, list[str]]


def citation(hit: RetrievalHit) -> Citation:
    if metadata_errors(hit):
        raise RenderError("Cannot cite incomplete or unpinned source metadata")
    leaf = (hit.section or "").split(" > ")[-1]
    number = re.match(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\b", leaf)
    section = number[1] if number is not None else leaf
    title = hit.title.split(":", 1)[0]
    pages = (
        f"p.{hit.page_start}"
        if hit.page_start == hit.page_end
        else f"pp.{hit.page_start}-{hit.page_end}"
    )
    return Citation(
        chunk_id=hit.chunk_id,
        paper_id=hit.paper_id,
        arxiv_id=hit.arxiv_id,
        arxiv_version=hit.arxiv_version,
        title=hit.title,
        section=hit.section or "",
        page_start=hit.page_start,
        page_end=hit.page_end,
        display=f"[{title} v{hit.arxiv_version}, §{section}, {pages}]",
    )


def render_answer(
    draft: Draft, supplied: list[RetrievalHit], validation: ValidationResult
) -> RenderedAnswer:
    current = validate_evidence(draft, supplied)
    if not (
        validation.status == "pass"
        and validation.deterministic_passed
        and validation.verifier_passed
        and current.deterministic_passed
        and validation.draft_digest == current.draft_digest
        and validation.evidence_digest == current.evidence_digest
    ):
        raise RenderError("Draft/evidence must match the completed validation")
    indices = [v.i for v in validation.verdicts]
    if (
        len(indices) != len(draft.claims)
        or set(indices) != set(range(len(draft.claims)))
        or any(v.v != "supported" for v in validation.verdicts)
    ):
        raise RenderError("Every claim requires exactly one supported verdict")
    hits = {h.chunk_id: h for h in supplied}
    references: dict[str, Citation] = {}
    lines, associations = [], {}
    for i, claim in enumerate(draft.claims):
        ids = list(dict.fromkeys(e.chunk_id for e in claim.evidence))
        markers = []
        for identifier in ids:
            references.setdefault(identifier, citation(hits[identifier]))
            if references[identifier].display not in markers:
                markers.append(references[identifier].display)
        associations[i] = ids
        lines.append(claim.text + " " + " ".join(markers))
    return RenderedAnswer(
        text="\n\n".join(lines), citations=list(references.values()), claim_citations=associations
    )


def refusal_text(kind: Literal["local", "model", "validation"]) -> str:
    return {
        "local": "I couldn't find usable evidence for this question in the supplied papers.",
        "model": (
            "The supplied paper passages do not provide enough evidence to answer this question."
        ),
        "validation": "I couldn't verify a reliable answer from the available paper passages.",
    }[kind]
