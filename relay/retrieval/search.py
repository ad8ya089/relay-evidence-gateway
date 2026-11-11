"""BM25 and hybrid retrieval with explicit outcomes and stable provenance."""

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from relay.obs.clock import Clock, SystemClock
from relay.retrieval.models import Chunk, RetrievalHit, RetrievalResult
from relay.retrieval.rerank import DenseEncoder, FastEmbedEncoder, FastEmbedReranker, Reranker
from relay.settings import PaperId, RetrievalSettings, Setting, retrieval_settings


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _query_options(query: str) -> tuple[PaperId | None, bool]:
    # A lexical interpretation is a hint only. Explicit restriction is paper_filter.
    hint: PaperId | None = None
    if re.search(r"\bmoshi\b", query, re.I):
        hint = "moshi"
    elif re.search(r"dynamic chunking|h-net", query, re.I):
        hint = "dynamic_chunking"
    elif re.search(r"transformer|attention is all you need", query, re.I):
        hint = "transformer"
    numeric = bool(re.search(r"\d|how (?:many|much)|latency|complexity|milliseconds", query, re.I))
    return hint, numeric


def _explicit_paper(query: str, config: RetrievalSettings) -> PaperId | None:
    """Only an exact title or 'in/according to the ... paper' is a restriction.

    Ambiguous keyword guesses remain hints. Comparisons naming several papers
    remain unrestricted. The caller may always supply an explicit paper_filter.
    """
    mentioned = {
        paper.paper_id
        for paper in config.papers
        if re.search(rf"\b{re.escape(paper.paper_id.replace('_', ' '))}\b", query, re.I)
        or paper.title.casefold() in query.casefold()
    }
    if len(mentioned) > 1:
        return None
    named: set[PaperId] = set()
    for paper in config.papers:
        alias = paper.paper_id.replace("_", " ")
        if paper.title.casefold() in query.casefold() or re.search(
            rf"\b(?:in|according to|from)\s+(?:the\s+)?{re.escape(alias)}\s+paper\b",
            query,
            re.I,
        ):
            named.add(paper.paper_id)
    return next(iter(named)) if len(named) == 1 else None


class RetrievalEngine:
    def __init__(
        self,
        *,
        data_dir: Path = Path("data"),
        config: RetrievalSettings | None = None,
        fixture_chunks: dict[Setting, list[Chunk]] | None = None,
        dense_factory: Callable[[], DenseEncoder] | None = None,
        rerank_factory: Callable[[], Reranker] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config if config is not None else retrieval_settings()
        self.data_dir = data_dir
        self.clock = clock if clock is not None else SystemClock()
        self._fixture_chunks = fixture_chunks
        self._dense_factory = dense_factory or (
            lambda: FastEmbedEncoder(
                self.config.dense_model,
                self.data_dir / "indexes/models",
                threads=self.config.model_threads,
                batch_size=self.config.model_batch_size,
            )
        )
        self._rerank_factory = rerank_factory or (
            lambda: FastEmbedReranker(
                self.config.rerank_model,
                self.data_dir / "indexes/models",
                threads=self.config.model_threads,
                batch_size=self.config.model_batch_size,
            )
        )
        self._dense: DenseEncoder | None = None
        self._reranker: Reranker | None = None
        self._rerank_error: str | None = None
        self._chunks: dict[Setting, list[Chunk]] = {}
        self._lexical: dict[Setting, BM25Okapi | None] = {}
        self._embeddings: dict[str, NDArray[np.float32]] = {}
        self._lock = threading.Lock()

    def prepare_lexical(self, setting: Setting) -> list[Chunk]:
        if setting in self._chunks:
            return self._chunks[setting]
        if self._fixture_chunks is not None:
            chunks = self._fixture_chunks.get(setting, [])
        else:
            path = self.data_dir / "chunks" / f"{setting}.jsonl"
            chunks = [
                Chunk.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("duplicate chunk IDs in index")
        tokens = [tokenize(c.text) for c in chunks]
        self._lexical[setting] = BM25Okapi(tokens) if chunks and any(tokens) else None
        self._chunks[setting] = chunks
        if self._fixture_chunks is None:
            index_dir = self.data_dir / "indexes"
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / f"{setting}.lexical.json").write_text(
                json.dumps({"chunk_ids": [c.chunk_id for c in chunks], "tokens": tokens}),
                encoding="utf-8",
            )
        return chunks

    def _vectors(self, chunks: list[Chunk]) -> NDArray[np.float32]:
        if self._dense is None:
            self._dense = self._dense_factory()
        key = hashlib.sha256(
            (
                self.config.dense_model + "\n" + "\n".join(c.model_dump_json() for c in chunks)
            ).encode()
        ).hexdigest()
        if key not in self._embeddings:
            path = self.data_dir / "indexes" / f"dense-{key}.npy"
            if path.exists() and self._fixture_chunks is None:
                vectors = np.load(path, allow_pickle=False)
            else:
                vectors = self._dense.passages([c.text for c in chunks])
                if self._fixture_chunks is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".npy.part")
                    with temporary.open("wb") as handle:
                        np.save(handle, vectors, allow_pickle=False)
                    temporary.replace(path)
            if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
                raise ValueError("dense backend returned invalid passage embeddings")
            self._embeddings[key] = vectors.astype(np.float32)
        return self._embeddings[key]

    def prepare_dense(self, setting: Setting = "C") -> int:
        """Explicit batch preparation outside the per-query deadline.

        Used by the ingestion CLI; C and D share a content-addressed vector
        cache. Imports/installation/tests never call this automatically.
        """
        if self.config.settings[setting].method != "hybrid":
            raise ValueError("Dense preparation requires a hybrid setting")
        with self._lock:
            chunks = self.prepare_lexical(setting)
            if chunks:
                self._vectors(chunks)
            return len(chunks)

    def _rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if self._rerank_error is not None:
            raise RuntimeError(self._rerank_error)
        if self._reranker is None:
            self._reranker = self._rerank_factory()
        scores = self._reranker.scores(query, texts)
        if len(scores) != len(texts) or not np.isfinite(scores).all():
            raise ValueError("cross-encoder returned missing or malformed scores")
        return scores

    def _search(
        self,
        query: str,
        paper_filter: PaperId | None,
        top_k: int | None,
        setting: Setting,
        paper_hint: PaperId | None,
        numeric: bool | None,
    ) -> RetrievalResult:
        cfg = self.config.settings[setting]
        result = RetrievalResult(
            status="empty", setting=setting, query_used=query, method=cfg.method
        )
        inferred_hint, inferred_numeric = _query_options(query)
        hint = paper_hint if paper_hint is not None else inferred_hint
        if paper_filter is None:
            paper_filter = _explicit_paper(query, self.config)
        numeric = inferred_numeric if numeric is None else numeric
        limit = cfg.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        chunks = self.prepare_lexical(setting)
        terms = tokenize(query)
        bm25 = self._lexical[setting]
        if not terms or bm25 is None:
            return result
        indices = [
            i for i, c in enumerate(chunks) if paper_filter is None or c.paper_id == paper_filter
        ]
        if not indices:
            return result
        raw = bm25.get_scores(terms)
        # Zero scores for absent terms are not evidence. Keep matching negative
        # Okapi scores (possible on tiny/common-word corpora), ordered by raw score.
        lexical = [i for i in indices if set(terms).intersection(tokenize(chunks[i].text))]
        lexical.sort(key=lambda i: (-float(raw[i]), chunks[i].chunk_id))
        bm_ranks = {i: r for r, i in enumerate(lexical, 1)}
        dense_ranks: dict[int, int] = {}
        dense_scores: dict[int, float] = {}
        fusion: dict[int, float] = {}
        if cfg.method == "hybrid":
            vectors = self._vectors(chunks)
            assert self._dense is not None
            query_vector = self._dense.query(query)
            if query_vector.ndim != 1 or not np.isfinite(query_vector).all():
                raise ValueError("dense backend returned an invalid query embedding")
            denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
            similarity = np.divide(
                vectors @ query_vector,
                denominator,
                out=np.zeros(len(vectors), dtype=np.float32),
                where=denominator != 0,
            )
            dense_order = sorted(
                [i for i in indices if similarity[i] > self.config.dense_min_score],
                key=lambda i: (-float(similarity[i]), chunks[i].chunk_id),
            )
            dense_ranks = {i: rank for rank, i in enumerate(dense_order, 1)}
            dense_scores = {i: float(similarity[i]) for i in indices}
            for ranks in (bm_ranks, dense_ranks):
                for i, rank in ranks.items():
                    if rank <= cfg.candidates:
                        fusion[i] = fusion.get(i, 0.0) + 1 / (self.config.rrf_k + rank)
            scores = fusion.copy()
        else:
            # Shift only negative Okapi corpora so a positive hint multiplier can
            # never penalise the hinted paper. Original BM25 scores remain intact.
            floor = min((float(raw[i]) for i in lexical), default=0.0)
            scores = {i: float(raw[i]) - min(floor, 0.0) for i in lexical}
            if scores and max(scores.values()) == 0:
                scores = {i: 1 / bm_ranks[i] for i in lexical}
        for i in scores:
            if chunks[i].paper_id == hint:
                scores[i] *= self.config.paper_hint_boost
        ranked = sorted(scores, key=lambda i: (-scores[i], chunks[i].chunk_id))
        rerank_scores: dict[int, float] = {}
        if cfg.rerank and ranked:
            ranked = ranked[: cfg.candidates]
            try:
                values = self._rerank(query, [chunks[i].text for i in ranked])
                rerank_scores = dict(zip(ranked, values, strict=True))
                # Cross-encoder logits can be negative. Map them monotonically
                # through sigmoid before boosting, so x1.3 is always a boost.
                for i in ranked:
                    scores[i] = float(np.exp(-np.logaddexp(0.0, -rerank_scores[i])))
                    if chunks[i].paper_id == hint:
                        scores[i] *= self.config.paper_hint_boost
                ranked.sort(key=lambda i: (-scores[i], chunks[i].chunk_id))
                result.rerank_method = "cross_encoder"
            except Exception as exc:
                # External model boundary: ANY loading/inference failure falls
                # back explicitly. No retry, no substitute embedding model.
                self._rerank_error = self._rerank_error or type(exc).__name__
                result.diagnostics.append(
                    f"Cross-encoder unavailable ({self._rerank_error}); using RRF only"
                )
        final = ranked[: min(limit, cfg.top_k)]
        neighbours: set[int] = set()
        if numeric and final:
            top = final[0]
            source = chunks[top]
            same_section = [
                i
                for i, c in enumerate(chunks)
                if c.paper_id == source.paper_id
                and c.section == source.section
                and c.kind == source.kind
            ]
            position = same_section.index(top)
            # Include immediate previous AND next if present: either may hold the qualifier.
            for p in (position - 1, position + 1):
                if 0 <= p < len(same_section):
                    neighbour = same_section[p]
                    if neighbour not in final:
                        final.append(neighbour)
                        neighbours.add(neighbour)
            if neighbours:
                result.diagnostics.append("numeric context includes adjacent same-section chunks")
        result.hits = [
            RetrievalHit(
                **chunks[i].model_dump(),
                score=scores.get(i, 0.0),
                rank=rank,
                bm25_score=float(raw[i]),
                bm25_rank=bm_ranks.get(i),
                dense_score=dense_scores.get(i),
                dense_rank=dense_ranks.get(i),
                rrf_score=fusion.get(i),
                rerank_score=rerank_scores.get(i),
                hint_multiplier=self.config.paper_hint_boost if chunks[i].paper_id == hint else 1.0,
                neighbour=i in neighbours,
            )
            for rank, i in enumerate(final, 1)
        ]
        result.status = "ok" if result.hits else "empty"
        return result

    async def search_papers(
        self,
        query: str,
        paper_filter: PaperId | None = None,
        top_k: int | None = None,
        setting: Setting = "C",
        *,
        paper_hint: PaperId | None = None,
        numeric: bool | None = None,
    ) -> RetrievalResult:
        started = self.clock.monotonic_ns()
        cfg = self.config.settings[setting]

        def work() -> RetrievalResult:
            # Serialise lazy model/cache initialisation; timeouts include lock wait.
            with self._lock:
                return self._search(query, paper_filter, top_k, setting, paper_hint, numeric)

        try:
            result = await asyncio.wait_for(asyncio.to_thread(work), self.config.timeout_ms / 1000)
        except TimeoutError:
            result = RetrievalResult(
                status="timeout",
                setting=setting,
                query_used=query,
                method=cfg.method,
                diagnostics=["retrieval deadline exceeded; model worker may still be finishing"],
            )
        except Exception as exc:
            # Tool boundary converts external index/model failures into a typed outcome.
            result = RetrievalResult(
                status="error",
                setting=setting,
                query_used=query,
                method=cfg.method,
                diagnostics=[
                    f"Retrieval failed ({type(exc).__name__}); check index/model availability"
                ],
            )
        result.took_ms = (self.clock.monotonic_ns() - started) / 1_000_000
        return result


@lru_cache(maxsize=1)
def default_engine() -> RetrievalEngine:
    return RetrievalEngine()


async def search_papers(
    query: str,
    paper_filter: PaperId | None = None,
    top_k: int | None = None,
    setting: Setting = "C",
    *,
    paper_hint: PaperId | None = None,
    numeric: bool | None = None,
) -> RetrievalResult:
    return await default_engine().search_papers(
        query, paper_filter, top_k, setting, paper_hint=paper_hint, numeric=numeric
    )
