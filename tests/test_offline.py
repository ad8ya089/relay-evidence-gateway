import socket
from pathlib import Path

import pytest

from relay.retrieval.tokenizer import get_encoding


def test_http_connections_blocked_including_localhost() -> None:
    with socket.socket() as connection:
        with pytest.raises(AssertionError, match="Network is forbidden"):
            connection.connect(("127.0.0.1", 80))
    with pytest.raises(AssertionError, match="Network is forbidden"):
        socket.create_connection(("example.invalid", 443))


def test_bundled_tokenizer_is_o200k_without_cache_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    get_encoding.cache_clear()
    encoding = get_encoding()
    assert encoding.name == "o200k_base"
    assert len(encoding._mergeable_ranks) == 199998
    text = "FAKE Unicode 😀 中文."
    assert encoding.decode(encoding.encode(text)) == text
