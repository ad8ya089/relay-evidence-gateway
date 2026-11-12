from relay.providers.base import Message
from standin.cache import PrefixCache, prompt_tokens, static_prefix
from tests.standin_helpers import clock, profile


def test_prefix_cache_excludes_dynamic_content_and_is_exact() -> None:
    c, settings = clock(), profile().cache
    cache = PrefixCache(c)
    a = [
        Message(role="system", content="static exact prefix"),
        Message(role="user", content="Q1 <<chunk:a>>one<<end>>"),
    ]
    b = [a[0], Message(role="user", content="Q2 <<chunk:b>>two<<end>>")]
    first = cache.lookup("a", a, settings)
    assert not first.hit and first.cached_tokens == 0
    cache.record_accepted(first)
    cache.populate(first, settings)
    second = cache.lookup("a", b, settings)
    assert second.hit and second.key == first.key
    assert 0 < second.cached_tokens <= prompt_tokens(b)
    assert static_prefix(a) == static_prefix(b)
    assert not cache.lookup("b", b, settings).hit
    b[0] = Message(role="system", content="static exact prefix ")
    assert not cache.lookup("a", b, settings).hit
    c.advance_ms(settings.ttl_s.value * 1000)
    assert not cache.lookup("a", a, settings).hit


def test_cache_reset_disabled_threshold_and_bounded_size() -> None:
    c = clock()
    settings = profile(cache__max_entries=1).cache
    cache = PrefixCache(c)
    messages = [Message(role="developer", content="prefix")]
    first = cache.lookup("a", messages, settings)
    cache.populate(first, settings)
    cache.populate(cache.lookup("b", messages, settings), settings)
    assert cache.state()["entries"] == 1
    assert not cache.lookup("a", messages, settings).hit
    assert cache.lookup("b", messages, settings).hit
    assert cache.lookup("a", messages, profile(cache__enabled=False).cache).key is None
    assert cache.lookup("a", messages, profile(cache__minimum_tokens=1000).cache).key is None
    cache.reset()
    assert cache.state() == {"hits": 0, "misses": 0, "entries": 0}
