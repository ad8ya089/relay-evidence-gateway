"""Bounded deterministic protocol responses, not a language model or verifier."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from relay.providers.base import Message


class InvalidInput(ValueError):
    """Malformed draft/verifier input; never an implicit verification pass."""


class Evidence(BaseModel):
    chunk_id: str
    quote: str


class Claim(BaseModel):
    text: str
    evidence: list[Evidence]


class Draft(BaseModel):
    status: Literal["answer", "insufficient_evidence"]
    claims: list[Claim]
    note: str


class VerifyInput(BaseModel):
    i: int = Field(ge=0)
    claim: str = Field(min_length=1)
    quotes: list[str] = Field(min_length=1)


class Verdict(BaseModel):
    i: int
    v: Literal["supported", "unsupported"]
    why: str


class Verification(BaseModel):
    verdicts: list[Verdict]


def call_kind(messages: list[Message]) -> Literal["draft", "verify", "generic"]:
    static = "\n".join(m.content for m in messages if m.role in {"system", "developer"})
    draft, verify = "RELAY_DRAFT_V1" in static, "RELAY_VERIFY_V1" in static
    if draft and verify:
        raise InvalidInput("Ambiguous simulator protocol markers")
    return "draft" if draft else "verify" if verify else "generic"


def _bounded_quote(content: str) -> str:
    """Prefer an intact passage over PDF headings or a clipped list preamble."""
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.endswith(":"):
            continue
        if re.match(r"^\d+(?:\.\d+)*\s+", paragraph) and not paragraph.endswith((".", "!", "?")):
            continue
        for sentence in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", paragraph, re.S):
            quote = sentence[0].strip()
            if quote and len(quote.split()) <= 40:
                return quote
    sentence = re.match(r".+?(?:[.!?](?=\s|$)|$)", content, re.S)
    if sentence is None:
        return ""
    words = list(re.finditer(r"\S+", sentence[0]))
    return sentence[0][: words[min(40, len(words)) - 1].end()]


def draft_response(text: str) -> Draft:
    chunks = list(re.finditer(r"<<chunk:([^<>\s]+)>>([\s\S]*?)<<end>>", text))
    if text.count("<<chunk:") != len(chunks) or text.count("<<end>>") != len(chunks):
        raise InvalidInput("Malformed chunk delimiters")
    identifiers = [match[1] for match in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise InvalidInput("Duplicate chunk identifiers")
    for chunk in chunks:
        content = chunk[2].strip()
        if not content:
            continue
        quote = _bounded_quote(content)
        if not quote:
            continue
        return Draft(
            status="answer",
            claims=[Claim(text=quote, evidence=[Evidence(chunk_id=chunk[1], quote=quote)])],
            note="",
        )
    return Draft(status="insufficient_evidence", claims=[], note="No usable evidence was supplied.")


def verify_response(text: str, fail: bool) -> Verification:
    try:
        claims = TypeAdapter(list[VerifyInput]).validate_json(text)
    except ValidationError:
        raise InvalidInput(
            "Verifier expects a JSON list of indexed claims and supporting quotes"
        ) from None
    if (
        not claims
        or any(not c.claim.strip() for c in claims)
        or len({c.i for c in claims}) != len(claims)
        or any(not q.strip() for c in claims for q in c.quotes)
    ):
        raise InvalidInput("Verifier requires nonempty unique claims and quotes")
    return Verification(
        verdicts=[
            Verdict(
                i=c.i,
                v="unsupported" if fail and n == 0 else "supported",
                why="Injected simulator failure."
                if fail and n == 0
                else "Simulator accepts supplied claim and quotes.",
            )
            for n, c in enumerate(claims)
        ]
    )


def generate(messages: list[Message], *, verify_fail: bool = False) -> str:
    kind = call_kind(messages)
    text = "\n".join(m.content for m in messages if m.role == "user")
    if kind == "draft":
        return draft_response(text).model_dump_json()
    if kind == "verify":
        return verify_response(text, verify_fail).model_dump_json()
    return json.dumps({"answer": "ok"}, separators=(",", ":"))
