"""Manage process handles we created; never discover or kill processes by port."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from loadgen.service import ServiceSpec
from loadgen.settings import LoadSettings
from relay.obs.clock import Clock


class OwnedService:
    def __init__(
        self,
        spec: ServiceSpec,
        directory: Path,
        settings: LoadSettings,
        clock: Clock,
        on_port: Callable[[int], None] | None = None,
    ) -> None:
        self.spec, self.directory, self.settings, self.clock = spec, directory, settings, clock
        self.on_port = on_port
        self.process: asyncio.subprocess.Process | None = None
        self.url: str | None = None

    async def start(self) -> str:
        path = self.directory / f"{self.spec.role}_spec.json"
        path.write_text(self.spec.model_dump_json(indent=2), encoding="utf-8")
        with (self.directory / f"{self.spec.role}.log").open("wb") as log:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "loadgen.service",
                str(path),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        end = self.clock.monotonic_ns() + int(self.settings.startup_timeout_s * 1e9)
        try:
            while self.clock.monotonic_ns() < end:
                if self.process.returncode is not None:
                    raise RuntimeError(f"Owned {self.spec.role} exited; inspect its local log")
                if self.spec.ready.exists():
                    ready = json.loads(self.spec.ready.read_text())
                    if ready["run_id"] != self.spec.run_id or ready["role"] != self.spec.role:
                        raise ValueError("Service readiness ownership mismatch")
                    self.url = ready["url"]
                    if self.on_port:
                        self.on_port(urlsplit(self.url).port)
                    async with httpx.AsyncClient(
                        trust_env=False, timeout=self.settings.admin_timeout_s
                    ) as http:
                        endpoint = "/admin/state" if self.spec.role == "standin" else "/healthz"
                        response = await http.get(self.url + endpoint)
                        response.raise_for_status()
                    return self.url
                await asyncio.sleep(self.settings.health_poll_s)
            raise TimeoutError(f"Owned {self.spec.role} startup timed out")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        if self.url:
            try:
                async with httpx.AsyncClient(
                    trust_env=False, timeout=self.settings.admin_timeout_s
                ) as http:
                    await http.post(
                        self.url + "/__runner/stop", headers={"X-Relay-Run": self.spec.run_id}
                    )
            except httpx.TransportError:
                pass
        try:
            await asyncio.wait_for(process.wait(), self.settings.shutdown_timeout_s)
        except TimeoutError:
            if process.returncode is None:
                process.kill()  # Only this exact child process handle.
            await process.wait()
