"""Real ephemeral loopback Uvicorn, isolated from external network in tests."""

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import uvicorn
from fastapi import FastAPI


class LocalServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@asynccontextmanager
async def running(app: FastAPI, loopback_ports: set[int]) -> AsyncIterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    port = sock.getsockname()[1]
    loopback_ports.add(port)
    server = LocalServer(
        uvicorn.Config(
            app,
            access_log=False,
            log_level="critical",
            lifespan="off",
            proxy_headers=False,
            timeout_graceful_shutdown=1,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))

    async def ready() -> None:
        while not server.started:
            if task.done():
                await task
                raise AssertionError("Local Uvicorn terminated before startup")
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(ready(), 3)
        yield f"http://127.0.0.1:{port}/openai/v1"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 3)
        finally:
            sock.close()
            loopback_ports.discard(port)


async def until_idle(app: FastAPI) -> None:
    async def wait() -> None:
        while app.state.simulation.inflight:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 2)
