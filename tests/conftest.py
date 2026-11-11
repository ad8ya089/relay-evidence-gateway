"""All ordinary tests are offline, even if a developer has real credentials."""

import socket
import threading
from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Network is forbidden in ordinary unit tests")

    # Windows implements socketpair using an internal loopback connection.
    # Permit exactly that primitive so asyncio can create its wake-up pipe;
    # ordinary connections (including localhost) remain forbidden.
    native_pair = socket.socketpair
    native_connect = socket.socket.connect
    state = threading.local()

    def socketpair(*args: object, **kwargs: object) -> tuple[socket.socket, socket.socket]:
        state.internal_pair = True
        try:
            return native_pair(*args, **kwargs)
        finally:
            state.internal_pair = False

    def connect(sock: socket.socket, address: object) -> None:
        if getattr(state, "internal_pair", False):
            native_connect(sock, address)
        else:
            denied()

    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setenv("RELAY_ALLOW_REAL", "0")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
