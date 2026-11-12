"""Ephemeral HTTP fixtures only. These shapes are NOT claims about real Groq."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest


def completion() -> dict[str, object]:
    return {
        "id": "FAKE-completion",
        "model": "FAKE-model",
        "choices": [
            {"message": {"content": '{"answer":"FAKE evidence"}'}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 10},
            "queue_time": 0.001,
            "prompt_time": 0.002,
            "completion_time": 0.003,
        },
        "x_groq": {"id": "FAKE-provider-request"},
    }


class FakeGroqServer:
    def __init__(self) -> None:
        self.mode = "success"
        self.modes: list[str] = []
        self.request_count = 0
        self.payloads: list[dict[str, object]] = []
        self.paths: list[str] = []
        self.echo = "FAKE-echo-canary"
        self.disconnected = asyncio.Event()
        self.entered = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/openai/v1"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode("latin-1").split("\r\n")
            self.paths.append(lines[0].split()[1])
            length = next(
                int(line.split(":", 1)[1])
                for line in lines[1:]
                if line.lower().startswith("content-length:")
            )
            payload = json.loads(await reader.readexactly(length))
            self.payloads.append(payload)
            self.request_count += 1
            mode = (
                self.modes[min(self.request_count - 1, len(self.modes) - 1)]
                if self.modes
                else self.mode
            )
            self.entered.set()
            if mode == "hang":
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                await writer.drain()
                await reader.read()
                self.disconnected.set()
                return
            if payload.get("stream") and mode in {
                "success",
                "stream",
                "usage-final",
                "usage-only",
                "x-groq-usage",
                "no-usage",
                "malformed-stream",
                "disconnect",
                "incomplete",
                "empty-stream",
                "usage-before-disconnect",
            }:
                await self._stream(writer, mode)
                return
            status = {
                "minute": 429,
                "day": 429,
                "unknown": 429,
                "echo": 429,
                "server": 500,
                "unavailable": 503,
                "auth": 401,
                "created": 201,
            }.get(mode, 200)
            body = completion()
            if mode in {"minute", "day", "unknown", "echo", "server", "unavailable", "auth"}:
                body = {"error": {"code": f"FAKE_{mode.upper()}", "message": self.echo}}
            if mode == "missing-usage":
                body.pop("usage")
            if mode == "empty":
                body["choices"] = []
            content = b"malformed JSON" if mode == "malformed" else json.dumps(body).encode()
            headers = (
                f"HTTP/1.1 {status} FIXTURE\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(content)}\r\nConnection: close\r\n"
                "X-RateLimit-Limit-Requests: 1000\r\nX-RateLimit-Remaining-Requests: 999\r\n"
                "X-RateLimit-Limit-Tokens: 8000\r\nX-RateLimit-Remaining-Tokens: 7000\r\n"
                "X-Request-Id: FAKE-request\r\nAuthorization: Bearer FAKE-response-header\r\n"
                + ("Retry-After: 2\r\n" if status == 429 else "")
                + "\r\n"
            )
            writer.write(headers.encode() + content)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            self.disconnected.set()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self._tasks.discard(task)

    async def _stream(self, writer: asyncio.StreamWriter, mode: str) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n"
            b"X-RateLimit-Remaining-Tokens: 7000\r\n\r\n"
        )
        await writer.drain()

        async def frame(value: object, *, raw: bool = False) -> None:
            data = str(value) if raw else json.dumps(value)
            content = ("data: " + data + "\n\n").encode()
            writer.write(f"{len(content):x}\r\n".encode() + content + b"\r\n")
            await writer.drain()

        if mode != "empty-stream":
            await frame({"choices": [{"delta": {"reasoning": "FAKE reasoning"}}]})
            await frame({"choices": [{"delta": {"content": '{"answer":'}}]})
            if mode == "disconnect":
                writer.transport.abort()
                return
            if mode == "malformed-stream":
                await frame("{not JSON", raw=True)
            final = {
                "choices": [{"delta": {"content": '"FAKE evidence"}'}, "finish_reason": "stop"}]
            }
            if mode == "usage-final":
                final["usage"] = completion()["usage"]
            await frame(final)
            if mode not in {"no-usage", "usage-final"}:
                usage = completion()["usage"]
                await frame(
                    {"choices": [], "x_groq": {"usage": usage}}
                    if mode == "x-groq-usage"
                    else {"choices": [], "usage": usage}
                )
            if mode == "usage-before-disconnect":
                writer.transport.abort()
                return
        if mode != "incomplete":
            await frame("[DONE]", raw=True)
        writer.write(b"0\r\n\r\n")
        await writer.drain()


@pytest.fixture
async def fake_server(loopback_ports: set[int]) -> AsyncIterator[FakeGroqServer]:
    server = FakeGroqServer()
    await server.start()
    loopback_ports.add(server.port)
    try:
        yield server
    finally:
        await server.close()
        loopback_ports.discard(server.port)
