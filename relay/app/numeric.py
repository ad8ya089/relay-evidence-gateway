"""Conservative numeric anchors, units and local qualifiers, not semantic entailment."""

import re
from decimal import Decimal, DecimalException

from pydantic import BaseModel

from relay.app.models import DraftClaim
from relay.app.settings import PipelineSettings
from relay.app.text import normalize
from relay.retrieval.models import RetrievalHit

WORDS = {
    word: i
    for i, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        )
    )
}
NUMBER = re.compile(
    r"(?<![\w.])(?P<n>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|\b(?:"
    + "|".join(WORDS)
    + r")\b)"
    r"(?P<u>\s*(?:milliseconds?|ms|seconds?|s|percent|%|dimensions?|layers?|heads?|parameters?|tokens?|million|billion|thousand|[MBK]))?(?!\w)",
    re.I,
)
QUALIFIERS = {
    "mode": {
        "theoretical": r"\btheoretic(?:al|ally)?\b|\bin theory\b",
        "practical": (
            r"\bpractic(?:al|ally)?\b|\bin practice\b|\bmeasured\b|\bobserved\b|\bempirical\b"
        ),
    },
    "approximation": {"approximate": r"\bapproximately\b|\babout\b|\baround\b|\bapprox\.?|~"},
    "bound": {
        "upper_inclusive": r"\bat most\b|\bup to\b|<=|≤",
        "upper_strict": r"\bless than\b|(?<![<=>])<(?![=>])",
        "lower_inclusive": r"\bat least\b|>=|≥",
        "lower_strict": r"\bmore than\b|(?<![<=>])>(?![=>])",
    },
    "negation": {"negated": r"\bnot\b|\bnever\b"},
}
UNITS: dict[str, tuple[str, str]] = {
    "ms": ("time_ms", "1"),
    "millisecond": ("time_ms", "1"),
    "milliseconds": ("time_ms", "1"),
    "s": ("time_ms", "1000"),
    "second": ("time_ms", "1000"),
    "seconds": ("time_ms", "1000"),
    "%": ("percent", "1"),
    "percent": ("percent", "1"),
    "k": ("count", "1000"),
    "thousand": ("count", "1000"),
    "m": ("count", "1000000"),
    "million": ("count", "1000000"),
    "b": ("count", "1000000000"),
    "billion": ("count", "1000000000"),
}


class Fact(BaseModel):
    start: int
    end: int
    value: Decimal
    unit: str
    qualifiers: dict[str, str]
    conditions: list[str]


def conditions(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"\b(?:when|if|provided that|assuming)\b[^;]*", text, re.I):
        condition = re.split(r",(?=\s)|(?<=[.!?])\s+", match[0])[0].strip(" .").casefold()
        found.append(condition)
    return found


def local_context(text: str, start: int, end: int, span: int) -> tuple[int, str]:
    left, right = max(0, start - span), min(len(text), end + span)
    boundaries = list(re.finditer(r"(?<=[.!?;])\s+", text))
    left = max([left] + [m.end() for m in boundaries if m.end() <= start])
    right = min([right] + [m.start() for m in boundaries if m.start() >= end])
    return left, text[left:right]


def qualifiers(
    text: str, start: int, end: int, spans: list[tuple[int, int]], context: int
) -> dict[str, str]:
    left, local = local_context(text, start, end, context)
    result = {}
    for category, variants in QUALIFIERS.items():
        candidates = []
        for label, expression in variants.items():
            for match in re.finditer(expression, local, re.I):
                a, b = left + match.start(), left + match.end()
                gap_left, gap_right = (b, start) if b <= start else (end, a)
                if gap_right < gap_left or any(
                    gap_left <= x and y <= gap_right for x, y in spans if (x, y) != (start, end)
                ):
                    continue
                candidates.append((gap_right - gap_left, label))
        if candidates:
            distance = min(c[0] for c in candidates)
            labels = {label for d, label in candidates if d == distance}
            result[category] = next(iter(labels)) if len(labels) == 1 else "ambiguous"
    return result


def facts(text: str, context: int) -> list[Fact]:
    matches = list(NUMBER.finditer(text))
    spans = [(m.start(), m.end()) for m in matches]
    result = []
    for match in matches:
        raw, unit = match["n"].casefold(), (match["u"] or "").strip().casefold()
        value = Decimal(WORDS[raw]) if raw in WORDS else Decimal(raw.replace(",", ""))
        canonical, scale = UNITS.get(unit, (unit.rstrip("s"), "1"))
        if not unit:
            canonical = "count"
        _, local = local_context(text, match.start(), match.end(), context)
        result.append(
            Fact(
                start=match.start(),
                end=match.end(),
                value=value * Decimal(scale),
                unit=canonical,
                qualifiers=qualifiers(text, match.start(), match.end(), spans, context),
                conditions=conditions(local),
            )
        )
    return result


def compatible(claim: Fact, quoted: Fact, source: Fact) -> bool:
    if not (
        claim.value == quoted.value == source.value and claim.unit == quoted.unit == source.unit
    ):
        return False
    if "ambiguous" in source.qualifiers.values():
        return False
    if claim.qualifiers != source.qualifiers or quoted.qualifiers != source.qualifiers:
        return False
    return (
        set(source.conditions).issubset(claim.conditions)
        and set(source.conditions).issubset(quoted.conditions)
        and set(claim.conditions).issubset(quoted.conditions)
    )


def numeric_errors(
    claim: DraftClaim, supplied: dict[str, RetrievalHit], config: PipelineSettings
) -> list[str]:
    text = normalize(claim.text)
    try:
        claimed = facts(text, config.numeric_context_chars)
        candidates: list[tuple[Fact, Fact]] = []
        for evidence in claim.evidence:
            source = normalize(supplied[evidence.chunk_id].text)
            quote = normalize(evidence.quote)
            qfacts = facts(quote, config.numeric_context_chars)
            sfacts = facts(source, config.numeric_context_chars)
            offset = source.find(quote)
            while offset >= 0:
                for qfact in qfacts:
                    for sfact in sfacts:
                        if sfact.start == offset + qfact.start:
                            candidates.append((qfact, sfact))
                offset = source.find(quote, offset + 1)
        if any(not any(compatible(fact, q, s) for q, s in candidates) for fact in claimed):
            return ["numeric_value_unit_or_qualifier_unsupported"]
        # Explicit mathematical comparison boundaries must also be preserved.
        relation = r"\b(?:[a-z]|\d+(?:\.\d+)?)\s*(?:<=|>=|<|>|≤|≥)\s*(?:[a-z]|\d+(?:\.\d+)?)\b"
        quote_text = " ".join(normalize(e.quote) for e in claim.evidence)
        relations = [re.sub(r"\s+", "", m[0]) for m in re.finditer(relation, text, re.I)]
        available = [re.sub(r"\s+", "", m[0]) for m in re.finditer(relation, quote_text, re.I)]
        if any(value not in available for value in relations):
            return ["comparison_boundary_unsupported"]
        if (claimed or re.search(r"complexity|latency", text, re.I)) and any(
            c not in conditions(text) for c in conditions(quote_text)
        ):
            return ["condition_omitted"]
    except (DecimalException, ValueError, KeyError):
        return ["numeric_validation_unavailable"]
    return []
