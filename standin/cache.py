"""Bounded model-aware static-prefix simulation; no request text in diagnostics."""

import hashlib
import json
import threading

from pydantic import BaseModel

from relay.obs.clock import Clock
from relay.providers.base import Message
from relay.retrieval.tokenizer import get_encoding
from standin.profile import CacheSettings


class CacheLookup(BaseModel):
    key: str | None
    prefix_tokens: int
    cached_tokens: int
    hit: bool


def static_prefix(messages: list[Message]) -> str:
    prefix = []
    for message in messages:
        if message.role not in {"system", "developer"}:
            break
        prefix.append(message.model_dump())
    return json.dumps(prefix, ensure_ascii=False, separators=(",", ":")) if prefix else ""


def prompt_tokens(messages: list[Message]) -> int:
    # Prefix uses identical role/content framing and is a subset of the full input estimate.
    return len(
        get_encoding().encode(
            json.dumps(
                [m.model_dump() for m in messages], ensure_ascii=False, separators=(",", ":")
            )
        )
    )


class PrefixCache:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, int] = {}
        self.hits, self.misses = 0, 0

    def lookup(self, model: str, messages: list[Message], settings: CacheSettings) -> CacheLookup:
        prefix = static_prefix(messages)
        tokens = len(get_encoding().encode(prefix)) if prefix else 0
        if not settings.enabled.value or not prefix or tokens < settings.minimum_tokens.value:
            return CacheLookup(key=None, prefix_tokens=tokens, cached_tokens=0, hit=False)
        key = hashlib.sha256((model + "\0" + prefix).encode()).hexdigest()
        with self._lock:
            now = self.clock.monotonic_ns()
            self._entries = {k: expires for k, expires in self._entries.items() if expires > now}
            hit = key in self._entries
            return CacheLookup(
                key=key, prefix_tokens=tokens, cached_tokens=tokens if hit else 0, hit=hit
            )

    def record_accepted(self, lookup: CacheLookup) -> None:
        with self._lock:
            if lookup.hit:
                self.hits += 1
            else:
                self.misses += 1

    def populate(self, lookup: CacheLookup, settings: CacheSettings) -> None:
        if lookup.key is None:
            return
        with self._lock:
            while (
                len(self._entries) >= settings.max_entries.value and lookup.key not in self._entries
            ):
                self._entries.pop(next(iter(self._entries)))
            # TTL starts at completed generation, not a request that has not produced output.
            self._entries[lookup.key] = self.clock.monotonic_ns() + int(settings.ttl_s.value * 1e9)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def state(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": sum(
                    expires > self.clock.monotonic_ns() for expires in self._entries.values()
                ),
            }
