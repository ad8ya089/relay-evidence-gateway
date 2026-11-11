import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from relay.retrieval.models import Chunk
from relay.retrieval.search import RetrievalEngine
from relay.settings import Setting, retrieval_settings
from tests.fixtures.backends import FakeDense, FakeReranker, fake_chunk, fixture_engine


def index(chunks: list[Chunk], **kwargs: object) -> RetrievalEngine:
    return RetrievalEngine(
        fixture_chunks={name: chunks for name in retrieval_settings().settings},
        dense_factory=FakeDense,
        **kwargs,
    )


@pytest.mark.parametrize("setting", ["A", "B"])
async def test_unrelated_is_explicit_empty_and_bm25(setting: Setting) -> None:
    result = await fixture_engine().search_papers("zxqvplm", setting=setting)
    assert result.status == "empty"
    assert result.model_dump()["hits"] == []
    assert result.method == "bm25" and result.rerank_method == "none"


@pytest.mark.parametrize("setting", ["C", "D"])
async def test_no_lexical_or_dense_signal_is_explicit_empty(setting: Setting) -> None:
    result = await fixture_engine().search_papers("zxqvplm", setting=setting)
    assert result.status == "empty" and result.hits == []
    assert result.method == "hybrid"


@pytest.mark.parametrize("setting", ["A", "B"])
async def test_lexical_settings_never_load_models(setting: Setting) -> None:
    def forbidden() -> FakeDense:
        raise AssertionError("BM25 must not load dense models")

    chunks = [fake_chunk(0, "FAKE alpha beta"), fake_chunk(1, "FAKE gamma delta", "moshi")]
    engine = RetrievalEngine(fixture_chunks={setting: chunks}, dense_factory=forbidden)
    result = await engine.search_papers("alpha", setting=setting)
    assert result.status == "ok" and result.method == "bm25"
    assert result.hits[0].bm25_rank == 1
    assert result.hits[0].dense_rank is None


@pytest.mark.parametrize("setting", ["C", "D"])
async def test_hybrid_path_and_original_ranks(setting: Setting) -> None:
    dense, reranker = FakeDense(), FakeReranker()
    chunks = [fake_chunk(i, f"FAKE alpha evidence {i}") for i in range(16)]
    engine = RetrievalEngine(
        fixture_chunks={"C": chunks, "D": chunks},
        dense_factory=lambda: dense,
        rerank_factory=lambda: reranker,
    )
    result = await engine.search_papers("alpha", setting=setting, numeric=False)
    assert result.status == "ok" and result.method == "hybrid"
    assert dense.passage_calls == 1 and dense.query_calls > 1
    assert len(result.hits) == 4
    assert all(h.bm25_rank and h.dense_rank and h.rrf_score for h in result.hits)
    if setting == "C":
        assert result.rerank_method == "cross_encoder"
        assert len(reranker.calls[0]) == 12
        assert all(h.rerank_score is not None for h in result.hits)
    else:
        assert result.rerank_method == "none" and not reranker.calls


async def test_explicit_dense_preparation_is_shared_by_c_and_d() -> None:
    dense = FakeDense()
    chunks = [fake_chunk(0, "FAKE alpha evidence"), fake_chunk(1, "FAKE beta evidence", "moshi")]
    engine = RetrievalEngine(
        fixture_chunks={"C": chunks, "D": chunks},
        dense_factory=lambda: dense,
        rerank_factory=FakeReranker,
    )
    assert await asyncio.to_thread(engine.prepare_dense, "C") == 2
    assert dense.passage_calls == 1
    await engine.search_papers("alpha", setting="C")
    await engine.search_papers("alpha", setting="D")
    assert dense.passage_calls == 1


async def test_missing_cross_encoder_is_explicit_rrf_fallback() -> None:
    def missing() -> FakeReranker:
        raise ImportError("FAKE unavailable")

    chunks = [fake_chunk(0, "FAKE alpha evidence"), fake_chunk(1, "FAKE beta", "moshi")]
    engine = index(chunks, rerank_factory=missing)
    result = await engine.search_papers("alpha", setting="C")
    ablation = await engine.search_papers("alpha", setting="D")
    assert result.status == "ok" and result.rerank_method == "none"
    assert "ImportError" in result.diagnostics[0]
    assert [h.chunk_id for h in result.hits] == [h.chunk_id for h in ablation.hits]


async def test_malformed_cross_encoder_is_not_silent_success() -> None:
    class Broken:
        def scores(self, query: str, texts: Sequence[str]) -> list[float]:
            return []

    result = await index([fake_chunk(0, "FAKE alpha")], rerank_factory=Broken).search_papers(
        "alpha", setting="C"
    )
    assert result.status == "ok" and result.rerank_method == "none"
    assert result.diagnostics


@pytest.mark.parametrize("setting", ["A", "B", "C", "D"])
async def test_hint_boosts_without_filtering(setting: Setting) -> None:
    chunks = [
        fake_chunk(0, "FAKE alpha useful", "transformer"),
        fake_chunk(1, "FAKE alpha useful", "moshi"),
        fake_chunk(2, "FAKE gamma unrelated", "dynamic_chunking"),
    ]
    engine = index(chunks, rerank_factory=FakeReranker)
    base = await engine.search_papers("alpha", setting=setting)
    boosted = await engine.search_papers("alpha", setting=setting, paper_hint="moshi")
    before = next(h for h in base.hits if h.paper_id == "moshi")
    after = next(h for h in boosted.hits if h.paper_id == "moshi")
    assert after.score == pytest.approx(before.score * 1.3)
    assert {h.paper_id for h in boosted.hits} >= {"transformer", "moshi"}
    filtered = await engine.search_papers("alpha", "moshi", setting=setting)
    assert {h.paper_id for h in filtered.hits} == {"moshi"}


async def test_numeric_adds_same_section_adjacent_chunks_only() -> None:
    chunks = [
        fake_chunk(0, "FAKE alpha theoretical 160"),
        fake_chunk(1, "FAKE practical qualifier"),
        fake_chunk(2, "FAKE other section", section="different section"),
        fake_chunk(3, "FAKE other paper", "moshi"),
    ]
    result = await index(chunks).search_papers("alpha", top_k=1, setting="A", numeric=True)
    assert len(result.hits) == 2
    assert result.hits[1].neighbour
    assert result.hits[1].section == result.hits[0].section
    assert result.hits[1].paper_id == "transformer"


async def test_exact_paper_title_filters_but_ambiguous_keywords_do_not() -> None:
    chunks = [
        fake_chunk(0, "FAKE alpha useful", "transformer"),
        fake_chunk(1, "FAKE alpha useful", "moshi"),
        fake_chunk(2, "FAKE gamma unrelated", "dynamic_chunking"),
    ]
    engine = index(chunks)
    explicit = await engine.search_papers("alpha in Attention Is All You Need", setting="A")
    assert {h.paper_id for h in explicit.hits} == {"transformer"}
    guessed = await engine.search_papers("alpha transformer", setting="A")
    assert {h.paper_id for h in guessed.hits} == {"transformer", "moshi"}
    comparison = await engine.search_papers(
        "alpha compare Attention Is All You Need and Moshi", setting="A"
    )
    assert {h.paper_id for h in comparison.hits} == {"transformer", "moshi"}


async def test_stable_order_and_source_metadata() -> None:
    engine = fixture_engine()
    first = await engine.search_papers("alpha", setting="D")
    second = await engine.search_papers("alpha", setting="D")
    assert first.hits == second.hits
    assert first.hits[0].arxiv_id == "1706.03762" and first.hits[0].arxiv_version == 7
    assert [h.rank for h in first.hits] == list(range(1, len(first.hits) + 1))


async def test_dense_failure_is_error_not_bm25_substitution() -> None:
    def missing() -> FakeDense:
        raise OSError("FAKE unavailable")

    engine = RetrievalEngine(
        fixture_chunks={"C": [fake_chunk(0, "FAKE alpha")]}, dense_factory=missing
    )
    result = await engine.search_papers("alpha", setting="C")
    assert result.status == "error" and result.hits == [] and result.diagnostics


async def test_missing_index_explicit_error(tmp_path: Path) -> None:
    result = await RetrievalEngine(data_dir=tmp_path).search_papers("alpha", setting="A")
    assert result.status == "error" and result.hits == []


async def test_timeout_and_injected_clock() -> None:
    release = threading.Event()

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0

        def monotonic_ns(self) -> int:
            self.now += 1_000_000
            return self.now

    def blocked() -> FakeDense:
        release.wait(timeout=1)
        return FakeDense()

    config = retrieval_settings().model_copy(update={"timeout_ms": 10})
    engine = RetrievalEngine(
        config=config,
        fixture_chunks={"C": [fake_chunk(0, "FAKE alpha")]},
        dense_factory=blocked,
        rerank_factory=FakeReranker,
        clock=FakeClock(),
    )
    try:
        result = await engine.search_papers("alpha", setting="C")
        assert result.status == "timeout" and result.took_ms == 1
    finally:
        release.set()
        await asyncio.to_thread(lambda: None)
