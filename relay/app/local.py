"""Owned ephemeral P3 service for the local smoke command, never a Relay API."""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn

from relay.app.settings import PipelineSettings
from standin.profile import server_settings
from standin.service import create_app


@asynccontextmanager
async def local_standin(config: PipelineSettings) -> AsyncIterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(profile="development"),
            log_level="critical",
            access_log=False,
            proxy_headers=False,
            lifespan="off",
            timeout_graceful_shutdown=int(server_settings().graceful_shutdown_s.value),
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))

    async def ready() -> None:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Local stand-in ended before startup")
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(ready(), config.local_startup_timeout_ms / 1000)
        yield f"http://127.0.0.1:{port}/openai/v1"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, config.local_shutdown_timeout_ms / 1000)
        finally:
            sock.close()
