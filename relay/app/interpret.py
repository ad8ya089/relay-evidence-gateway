"""Advisory local interpretation; retrieval/evidence decide answerability."""

import re

from relay.app.models import Interpretation
from relay.app.settings import PipelineSettings, pipeline_settings
from relay.retrieval.search import tokenize
from relay.settings import PaperId, retrieval_settings

ALIASES: dict[PaperId, str] = {
    "transformer": r"\btransformers?\b|attention is all you need|encoder.decoder attention",
    "moshi": r"\bmoshi\b|real.time (?:speech|spoken) dialogue|inner monologue",
    "dynamic_chunking": r"dynamic chunking|\bh.net\b|hierarchical sequence model(?:ing|ling)",
}


def interpret(
    text: str, *, paper_scope: PaperId | None = None, config: PipelineSettings | None = None
) -> Interpretation:
    config = config or pipeline_settings()
    mentioned = [
        paper for paper, expression in ALIASES.items() if re.search(expression, text, re.I)
    ]
    numeric = bool(
        re.search(
            r"\d|how (?:many|much|long|large)|latency|complexity|percent|dimension|"
            r"model size|parameter|\bcounts?\b|milliseconds?",
            text,
            re.I,
        )
    )
    comparison = bool(
        re.search(r"\bcompar(?:e|ison)|\bversus\b|\bvs\.?\b|differ|between", text, re.I)
    )
    limitation = bool(re.search(r"limit(?:ation)?s?|weakness|drawback|failure|caveat", text, re.I))
    out_of_scope = bool(
        re.search(
            r"\bgroq\b|\bvaani\b|my account|our production|current (?:rate|quota)|"
            r"(?:recommend|best).*\brag\b|\brag\b.*(?:chunk size|recommend)",
            text,
            re.I,
        )
    )
    qtype = (
        "absent"
        if out_of_scope
        else "compare"
        if comparison
        else "limitation"
        if limitation
        else "number"
        if numeric
        else "explain"
    )
    explicit = paper_scope
    if explicit is None and len(mentioned) == 1:
        paper = mentioned[0]
        spec = next(p for p in retrieval_settings().papers if p.paper_id == paper)
        short = {
            "transformer": "transformer",
            "moshi": "moshi",
            "dynamic_chunking": r"dynamic chunking|h.net",
        }[paper]
        if spec.title.casefold() in text.casefold() or re.search(
            rf"(?:in|from|according to)\s+(?:the\s+)?(?:{short})\s+paper\b", text, re.I
        ):
            explicit = paper
    keywords = list(dict.fromkeys(word for word in tokenize(text) if word not in config.stopwords))
    return Interpretation(
        paper_hint=mentioned[0] if len(mentioned) == 1 else None,
        explicit_paper=explicit,
        mentioned_papers=mentioned,
        qtype=qtype,
        numeric=numeric,
        looks_out_of_scope=out_of_scope,
        keywords=keywords,
        diagnostics=["Scope heuristic is advisory"] if out_of_scope else [],
    )
