"""FAKE deterministic vectors/scores for test plumbing, never model substitutes."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from relay.retrieval.models import Chunk
from relay.retrieval.search import RetrievalEngine, tokenize
from relay.retrieval.tokenizer import get_encoding
from relay.settings import PaperId, Setting, retrieval_settings


def fake_chunk(
    index: int,
    text: str,
    paper_id: PaperId = "transformer",
    *,
    section: str = "FAKE section",
    page: int = 1,
) -> Chunk:
    config = retrieval_settings()
    paper = next(p for p in config.papers if p.paper_id == paper_id)
    return Chunk(
        **paper.model_dump(),
        chunk_id=f"{paper_id}-v{paper.arxiv_version}-p{page}-c{index}",
        section=section,
        page_start=page,
        page_end=page,
        kind="text",
        text=text,
        n_tokens=len(get_encoding().encode(text)),
    )


class FakeDense:
    def __init__(self) -> None:
        self.passage_calls = 0
        self.query_calls = 0

    def query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        terms = tokenize(text)
        return np.asarray([terms.count(word) for word in ("alpha", "beta", "gamma")], np.float32)

    def passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.passage_calls += 1
        return np.asarray([self.query(text) for text in texts], dtype=np.float32)


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(list(texts))
        terms = set(tokenize(query))
        return [float(len(terms.intersection(tokenize(text)))) for text in texts]


def fixture_engine() -> RetrievalEngine:
    chunks = [
        fake_chunk(0, "FAKE alpha evidence"),
        fake_chunk(1, "FAKE beta evidence", "moshi"),
        fake_chunk(2, "FAKE gamma unrelated", "dynamic_chunking"),
    ]
    settings: dict[Setting, list[Chunk]] = {name: chunks for name in retrieval_settings().settings}
    return RetrievalEngine(
        fixture_chunks=settings, dense_factory=FakeDense, rerank_factory=FakeReranker
    )
