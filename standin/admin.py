"""Loopback-only validated administration; no request contents in state snapshots."""

import ipaddress
from collections import deque

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from relay.obs.events import Sanitizer
from standin.faults import Sampler
from standin.profile import Profile, load_profile
from standin.simulation import Simulation


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    profile: Profile | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> "ProfileUpdate":
        if (self.name is None) == (self.profile is None):
            raise ValueError("Supply either a configured profile name or a complete profile")
        return self


class LimitsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)
    tpd: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def nonempty(self) -> "LimitsUpdate":
        if all(getattr(self, name) is None for name in ("rpm", "tpm", "rpd", "tpd")):
            raise ValueError("Supply at least one quota update")
        return self


def local_only(request: Request) -> None:
    try:
        allowed = (
            request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise HTTPException(status_code=403, detail="Simulator administration requires loopback")


def admin_router(simulation: Simulation) -> APIRouter:
    router = APIRouter(prefix="/admin")

    def updated(action: str) -> dict[str, object]:
        simulation.last_admin_update = {
            "action": action,
            "observed_ns": simulation.clock.monotonic_ns(),
            "utc": simulation.clock.utc_now().isoformat(),
        }
        return Sanitizer().value(simulation.snapshot())

    @router.get("/state")
    async def state(request: Request) -> dict[str, object]:
        local_only(request)
        return Sanitizer().value(simulation.snapshot())

    @router.post("/limits")
    async def limits(payload: LimitsUpdate, request: Request) -> dict[str, object]:
        local_only(request)
        with simulation.lock:
            data = simulation.profile.model_dump()
            model = payload.model or simulation.profile.model
            if model not in data["models"]:
                raise HTTPException(status_code=400, detail="Model is not configured")
            for name in ("rpm", "tpm", "rpd", "tpd"):
                if (value := getattr(payload, name)) is not None:
                    data["models"][model][name] = {
                        "value": value,
                        "source": "chosen",
                        "note": "Hot simulator administrative override",
                    }
            replacement = Profile.model_validate(data).runnable()
            simulation.profile = replacement
            return updated("limits")

    @router.post("/profile")
    async def profile(payload: ProfileUpdate, request: Request) -> dict[str, object]:
        local_only(request)
        try:
            replacement = load_profile(payload.name) if payload.name else payload.profile.runnable()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        with simulation.lock:
            if simulation.inflight:
                raise HTTPException(
                    status_code=409, detail="Profile replacement requires quiescence"
                )
            simulation.profile = replacement
            simulation.sampler = Sampler(replacement)
            simulation.decisions = deque(maxlen=replacement.diagnostic_history.value)
            simulation.cache.reset()
            return updated("profile")

    @router.post("/reset")
    async def reset(request: Request) -> dict[str, object]:
        local_only(request)
        with simulation.lock:
            if simulation.inflight:
                raise HTTPException(
                    status_code=409, detail="Cannot reset while requests are active"
                )
            simulation.quotas.reset()
            simulation.cache.reset()
            simulation.sampler.reset()
            simulation.stats.clear()
            simulation.fault_counts.clear()
            simulation.decisions.clear()
            return updated("reset")

    return router
