from pathlib import Path

import pytest

from relay.app.cache import AnswerCache, cache_key, for_request
from relay.app.delivery import iter_validated_answer
from relay.app.models import State
from relay.control.settings import load_policy
from tests.fixtures.app_provider import verdicts
from tests.test_app_orchestrator import setup
from tests.test_app_validation import example


@pytest.mark.parametrize(
    "field", ["setting", "model", "prompt_version", "corpus_hash", "paper_scope"]
)
def test_cache_identity_is_complete_and_normalized(field: str) -> None:
    args = {
        "question": " FAKE   Question ",
        "setting": "A",
        "model": "FAKE",
        "prompt_version": "v1",
        "corpus_hash": "abc",
        "paper_scope": None,
    }
    assert cache_key(**args) == cache_key(**(args | {"question": "fake question"}))
    assert cache_key(**args) != cache_key(**(args | {field: "changed"}))


async def test_cache_preserves_seal_citations_deep_copy_persistence_and_ttl(tmp_path: Path) -> None:
    draft, _ = await example()
    app, _ = await setup([draft.model_dump_json(), verdicts("supported")])
    outcome = await app.handle("alpha", setting="A")
    config = load_policy().cache.model_copy(update={"enabled": True, "ttl_s": 1})
    cache = AnswerCache(tmp_path / "cache", config, app.clock)
    assert await cache.put("key", outcome)
    cached = await cache.get("key")
    cached.citations[0].title = "FAKE mutation"
    assert (await cache.get("key")).citations[0].title == outcome.citations[0].title
    cache.close()
    cache = AnswerCache(tmp_path / "cache", config, app.clock)
    cached = await cache.get("key")
    response = for_request(cached, "new", 0, 10**12, cached=True)
    assert response.state == State.CACHE_HIT and response.provider_calls == 0
    assert (
        "".join([p async for p in iter_validated_answer(response, clock=app.clock)]) == outcome.text
    )
    app.clock.advance_ms(1000)
    assert await cache.get("key") is None
    cache.close()


async def test_unsealed_or_failed_outcomes_cannot_populate_cache(tmp_path: Path) -> None:
    app, _ = await setup([])
    outcome = await app.handle("zqxv", setting="A")
    cache = AnswerCache(
        tmp_path / "cache", load_policy().cache.model_copy(update={"enabled": True}), app.clock
    )
    assert await cache.put("safe-refusal", outcome)
    outcome.text = "FAKE unverified output"
    assert not await cache.put("tampered", outcome)
    outcome.state = State.TIMED_OUT
    assert not await cache.put("timeout", outcome)
    assert await cache.get("timeout") is None
    cache.close()
