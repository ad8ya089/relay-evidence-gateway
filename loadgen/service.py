"""Single-worker, ephemeral-port child services owned by one experiment runner."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import HTTPException, Request

from loadgen.settings import Model, load_settings
from relay.app.api import create_local_app
from standin.admin import local_only
from standin.profile import Profile
from standin.service import create_app


class ServiceSpec(Model):
    role: Literal["standin", "relay"]
    run_id: str
    ready: Path
    profile: dict[str, object]
    policy_path: Path
    events_path: Path
    ledger_path: Path
    cache_path: Path
    base_url: str | None = None


async def serve(spec: ServiceSpec) -> None:
    settings = load_settings()
    if spec.role == "standin":
        app = create_app(profile=Profile.model_validate(spec.profile))
    else:
        app = create_local_app(
            policy_path=spec.policy_path,
            base_url=spec.base_url,
            ledger_path=spec.ledger_path,
            cache_path=spec.cache_path,
            events_path=spec.events_path,
            allow_unsafe_naive=True,
            run_id=spec.run_id,
            model=str(spec.profile["model"]),
        )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            workers=1,
            log_level="warning",
            access_log=False,
            proxy_headers=False,
            timeout_graceful_shutdown=settings.shutdown_timeout_s,
        )
    )

    @app.post("/__runner/stop")
    async def stop(request: Request) -> dict[str, bool]:
        local_only(request)
        if request.headers.get("x-relay-run") != spec.run_id:
            raise HTTPException(status_code=403, detail="Run ownership mismatch")
        server.should_exit = True
        return {"stopping": True}

    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Service exited before readiness")
            await asyncio.sleep(settings.health_poll_s)
        port = server.servers[0].sockets[0].getsockname()[1]
        temporary = spec.ready.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"run_id": spec.run_id, "role": spec.role, "url": f"http://127.0.0.1:{port}"}
            ),
            encoding="utf-8",
        )
        temporary.replace(spec.ready)
        await task
    finally:
        server.should_exit = True
        if not task.done():
            await task


if __name__ == "__main__":
    asyncio.run(serve(ServiceSpec.model_validate_json(Path(sys.argv[1]).read_text())))
