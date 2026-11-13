"""Evidence confidence and meaningful bounded query plans; weak is not absence."""

import math
from typing import Literal

from pydantic import BaseModel

from relay.app.models import Interpretation
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.retrieval.models import RetrievalHit, RetrievalResult
from relay.retrieval.search import tokenize
from relay.settings import PaperId


class GateDecision(BaseModel):
    status: Literal["sufficient", "weak", "empty"]
    keyword_coverage: float
    matching_terms: int
    score: float | None
    source: Literal["assumed"] = "assumed"


class SearchPlan(BaseModel):
    query: str
    paper_filter: PaperId | None
    paper_hint: PaperId | None
    numeric: bool
    reason: str


class EvidenceConflict(ValueError):
    """Retrieval reused an ID with inconsistent source-owned metadata/content."""


def evaluate(
    result: RetrievalResult, interpretation: Interpretation, config: PipelineSettings | None = None
) -> GateDecision:
    config = config or pipeline_settings()
    if result.status not in {"ok", "empty"}:
        raise ValueError("Tool failure cannot become an evidence refusal")
    if not result.hits:
        return GateDecision(status="empty", keyword_coverage=0, matching_terms=0, score=None)
    keywords = set(interpretation.keywords)
    terms = set().union(*(set(tokenize(hit.text)) for hit in result.hits))
    matching = len(keywords.intersection(terms))
    coverage = matching / len(keywords) if keywords else 0
    score = max(hit.score for hit in result.hits)
    sufficient = (
        math.isfinite(score)
        and score > config.gate_min_score
        and coverage >= config.gate_min_keyword_coverage
        and matching >= config.gate_min_matching_terms
    )
    return GateDecision(
        status="sufficient" if sufficient else "weak",
        keyword_coverage=coverage,
        matching_terms=matching,
        score=score,
    )


def broadening_plan(
    question: str,
    interpretation: Interpretation,
    result: RetrievalResult,
    step: int,
    used_queries: list[str],
    config: PipelineSettings | None = None,
) -> SearchPlan | None:
    config = config or pipeline_settings()
    if step not in range(1, config.broaden_max + 1):
        return None
    known_domain = bool(
        result.hits or interpretation.mentioned_papers or interpretation.explicit_paper
    )
    section_words = list(
        dict.fromkeys(
            word
            for hit in result.hits
            for word in tokenize(hit.section or "")
            if word not in config.stopwords and not word.isdigit()
        )
    )
    if step == 1:
        extras = section_words + (
            config.section_terms[interpretation.qtype] if known_domain else []
        )
        extras = [word for word in dict.fromkeys(extras) if word not in tokenize(question)]
        if not extras:
            return None
        query = question + " " + " ".join(extras[: config.broadening_term_max])
        hint = interpretation.paper_hint
    else:
        identity = {"moshi", "transformer", "transformers", "dynamic", "chunking", "h", "net"}
        words = interpretation.keywords
        if interpretation.explicit_paper is None:
            words = [word for word in words if word not in identity]
        if not words:
            return None
        query = " ".join(words)
        hint = interpretation.paper_hint if interpretation.explicit_paper is not None else None
    if tuple(tokenize(query)) in {tuple(tokenize(q)) for q in used_queries}:
        return None
    return SearchPlan(
        query=query,
        paper_filter=interpretation.explicit_paper,
        paper_hint=hint,
        numeric=interpretation.numeric,
        reason="section_terms" if step == 1 else "keyword_query_and_numeric_neighbours",
    )


def _source_identity(hit: RetrievalHit) -> tuple[object, ...]:
    return tuple(
        getattr(hit, field)
        for field in (
            "paper_id",
            "arxiv_id",
            "arxiv_version",
            "title",
            "url",
            "section",
            "page_start",
            "page_end",
            "text",
            "kind",
        )
    )


def merge_results(previous: RetrievalResult, current: RetrievalResult) -> RetrievalResult:
    if previous.setting != current.setting or any(
        r.status not in {"ok", "empty"} for r in (previous, current)
    ):
        raise EvidenceConflict("Cannot merge conflicting settings or failed retrieval operations")
    seen = {hit.chunk_id: hit for hit in previous.hits}
    for hit in current.hits:
        if hit.chunk_id in seen and _source_identity(hit) != _source_identity(seen[hit.chunk_id]):
            raise EvidenceConflict("Chunk ID changed source identity during broadening")
        seen.setdefault(hit.chunk_id, hit)
    return current.model_copy(
        update={
            "status": "ok" if seen else "empty",
            "hits": list(seen.values()),
            "took_ms": previous.took_ms + current.took_ms,
            "diagnostics": previous.diagnostics
            + current.diagnostics
            + ["Broadening preserved prior evidence"],
        }
    )
