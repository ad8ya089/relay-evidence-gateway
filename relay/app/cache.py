"""Bounded validated-outcome cache; SQLite is trusted local application storage."""

import asyncio
import hashlib
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path

from pydantic import BaseModel

from relay.app.delivery import DeliveryError, _authorize, _fingerprint, approved
from relay.app.models import Outcome, State, TokenTotals
from relay.app.text import normalize
from relay.control.settings import CacheSettings
from relay.obs.clock import Clock


def cache_key(
    question: str,
    setting: str,
    model: str,
    prompt_version: str,
    corpus_hash: str,
    *,
    paper_scope: str | None = None,
) -> str:
    # JSON framing avoids collisions from delimiters embedded in questions.
    identity = [
        normalize(question).casefold(),
        setting,
        model,
        prompt_version,
        corpus_hash,
        paper_scope,
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def cacheable(outcome: Outcome) -> bool:
    return (
        approved(outcome)
        and outcome.status == "ok"
        and outcome.state in {State.RELEASED, State.REFUSED_LOCAL, State.REFUSED_MODEL}
    )


def for_request(
    source: Outcome,
    request_id: str,
    arrival_ns: int,
    deadline_ns: int,
    *,
    cached: bool = False,
    coalesced: bool = False,
) -> Outcome:
    if not approved(source):
        raise DeliveryError("Cannot copy an unapproved outcome")
    result = source.model_copy(deep=True)
    result.request_id = request_id
    result.arrival_ns, result.deadline_ns = arrival_ns, deadline_ns
    result.first_user_token_at = result.first_token_ms = None
    result.cache_hit, result.coalesced = cached, coalesced
    if cached or coalesced:
        result.calls = []
        result.usage = TokenTotals()
        result.draft_calls = result.verifier_calls = result.repair_calls = result.retries = (
            result.tool_retries
        ) = 0
        result.provider_ms = result.total_ms = 0
        result.provider_ttft_ms = None
        result.stage_ms = {}
        if cached:
            result.state = State.CACHE_HIT
            result.states = [State.ARRIVED, State.CACHE_HIT]
    _authorize(result)
    return result


class Entry(BaseModel):
    outcome: Outcome
    created_ns: int
    expires_ns: int


class AnswerCache:
    def __init__(self, path: Path, config: CacheSettings, clock: Clock) -> None:
        self.config, self.clock = config, clock
        self.lock = asyncio.Lock()
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.hits = self.misses = self.writes = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "seal TEXT NOT NULL, checksum TEXT NOT NULL, created_ns INTEGER NOT NULL, "
            "expires_ns INTEGER NOT NULL)"
        )

    def close(self) -> None:
        self.db.close()

    async def put(self, key: str, outcome: Outcome) -> bool:
        if not self.config.enabled or not cacheable(outcome):
            return False
        async with self.lock:
            clone = outcome.model_copy(deep=True)
            now = self.clock.monotonic_ns()
            entry = Entry(
                outcome=clone, created_ns=now, expires_ns=now + int(self.config.ttl_s * 1e9)
            )
            data = clone.model_dump(mode="json") | {
                "arrival_ns": clone.arrival_ns,
                "deadline_ns": clone.deadline_ns,
            }
            payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
            self.db.execute(
                "INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?)",
                (
                    key,
                    payload,
                    clone._delivery_seal,
                    hashlib.sha256(payload.encode()).hexdigest(),
                    entry.created_ns,
                    entry.expires_ns,
                ),
            )
            self.entries[key] = entry
            self.entries.move_to_end(key)
            self.writes += 1
            self.db.execute("DELETE FROM answers WHERE expires_ns <= ?", (now,))
            rows = self.db.execute(
                "SELECT key FROM answers ORDER BY created_ns DESC, key LIMIT -1 OFFSET ?",
                (self.config.max_entries,),
            ).fetchall()
            for (old,) in rows:
                self.db.execute("DELETE FROM answers WHERE key=?", (old,))
                self.entries.pop(old, None)
            while len(self.entries) > self.config.max_entries:
                self.entries.popitem(last=False)
            return True

    async def get(self, key: str) -> Outcome | None:
        if not self.config.enabled:
            return None
        async with self.lock:
            now = self.clock.monotonic_ns()
            entry = self.entries.get(key)
            if entry is None:
                row = self.db.execute(
                    "SELECT payload,seal,checksum,created_ns,expires_ns FROM answers WHERE key=?",
                    (key,),
                ).fetchone()
                if row and row[3] <= now < row[4]:
                    try:
                        if hashlib.sha256(row[0].encode()).hexdigest() != row[2]:
                            raise ValueError("Cache checksum mismatch")
                        restored = Outcome.model_validate_json(row[0])
                        if _fingerprint(restored) != row[1]:
                            raise ValueError("Cache approval mismatch")
                        restored._delivery_seal = row[1]
                        if not cacheable(restored):
                            raise ValueError("Unsafe cached outcome")
                        entry = Entry(outcome=restored, created_ns=row[3], expires_ns=row[4])
                    except ValueError:
                        self.db.execute("DELETE FROM answers WHERE key=?", (key,))
            if (
                entry is None
                or not entry.created_ns <= now < entry.expires_ns
                or not cacheable(entry.outcome)
            ):
                self.entries.pop(key, None)
                self.misses += 1
                return None
            self.entries[key] = entry
            self.entries.move_to_end(key)
            while len(self.entries) > self.config.max_entries:
                self.entries.popitem(last=False)
            self.hits += 1
            return entry.outcome.model_copy(deep=True)

    async def snapshot(self) -> dict[str, int | bool]:
        async with self.lock:
            return {
                "enabled": self.config.enabled,
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "memory_entries": len(self.entries),
            }
