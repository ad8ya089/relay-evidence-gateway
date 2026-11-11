"""Lazy adapters for installed fastembed APIs; no import-time downloads."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class DenseEncoder(Protocol):
    def passages(self, texts: Sequence[str]) -> NDArray[np.float32]: ...
    def query(self, text: str) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    def scores(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedEncoder:
    def __init__(self, model_name: str, cache_dir: Path, *, threads: int, batch_size: int) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)
        self.batch_size = batch_size

    def passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            list(self.model.passage_embed(list(texts), batch_size=self.batch_size)),
            dtype=np.float32,
        )

    def query(self, text: str) -> NDArray[np.float32]:
        return np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: Path, *, threads: int, batch_size: int) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"Configured cross-encoder is not supported: {model_name}")
        self.model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir), threads=threads
        )
        self.batch_size = batch_size

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        return [
            float(score)
            for score in self.model.rerank(query, list(texts), batch_size=self.batch_size)
        ]
