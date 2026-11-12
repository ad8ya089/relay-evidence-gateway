"""All ordinary tests are offline, even if a developer has real credentials."""

import asyncio
import socket
import threading
from typing import NoReturn

import pytest

pytest_plugins = ["tests.fake_groq_server"]


@pytest.fixture
def loopback_ports() -> set[int]:
    """Only ports allocated by an explicit fake-server fixture may be contacted."""
    return set()


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch, loopback_ports: set[int]) -> None:
    def denied(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Network is forbidden in ordinary unit tests")

    # Windows implements socketpair using an internal loopback connection.
    # Permit exactly that primitive so asyncio can create its wake-up pipe;
    # ordinary connections (including localhost) remain forbidden.
    native_pair = socket.socketpair
    native_connect = socket.socket.connect
    native_connect_ex = socket.socket.connect_ex
    native_create = socket.create_connection
    native_resolve = socket.getaddrinfo
    native_async_connect = asyncio.BaseEventLoop.create_connection
    state = threading.local()

    def allowed(address: object) -> bool:
        return (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] in {"127.0.0.1", b"127.0.0.1"}
            and address[1] in loopback_ports
        )

    def socketpair(*args: object, **kwargs: object) -> tuple[socket.socket, socket.socket]:
        state.internal_pair = True
        try:
            return native_pair(*args, **kwargs)
        finally:
            state.internal_pair = False

    def connect(sock: socket.socket, address: object) -> None:
        if getattr(state, "internal_pair", False) or allowed(address):
            native_connect(sock, address)
        else:
            denied()

    def connect_ex(sock: socket.socket, address: object) -> int:
        if allowed(address):
            return native_connect_ex(sock, address)
        denied()

    def create(address: object, *args: object, **kwargs: object) -> socket.socket:
        if allowed(address):
            return native_create(address, *args, **kwargs)
        denied()

    def resolve(host: object, port: object, *args: object, **kwargs: object) -> object:
        if allowed((host, port)):
            return native_resolve(host, port, *args, **kwargs)
        denied()

    async def async_connect(
        loop: asyncio.BaseEventLoop,
        factory: object,
        host: object = None,
        port: object = None,
        **kwargs: object,
    ) -> object:
        if not allowed((host, port)):
            denied()
        return await native_async_connect(loop, factory, host, port, **kwargs)

    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", async_connect)
    monkeypatch.setenv("RELAY_ALLOW_REAL", "0")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
