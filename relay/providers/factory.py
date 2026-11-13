"""Explicit application provider selection; credentials are read only after permission."""

import os
from pathlib import Path

from relay.control.settings import Policy, load_policy
from relay.providers.groq import is_loopback_url
from relay.providers.guard import RealCallDenied
from relay.settings import SafetySettings


def selection(policy: Policy, *, real: bool, base_url: str | None) -> tuple[str, str | None]:
    safety = SafetySettings()
    selected = os.environ.get("RELAY_PROVIDER", policy.provider)
    if selected == "standin":
        if real or policy.provider == "groq":
            raise RealCallDenied("Provider selection disagrees with explicit execution mode")
        url = base_url or policy.server.standin_base_url
        if not is_loopback_url(url):
            raise RealCallDenied("Stand-in selection requires loopback")
        return url, None
    if selected != "groq" or safety.provider != "groq" or safety.allow_real != "1" or not real:
        raise RealCallDenied("Groq requires RELAY_PROVIDER=groq, RELAY_ALLOW_REAL=1 and --real")
    url = base_url or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    if url.rstrip("/") != "https://api.groq.com/openai/v1":
        raise RealCallDenied("Real application requires the explicit official TLS Groq origin")
    key = os.environ.get("GROQ_API_KEY")
    if not key or not key.strip() or any(c in key for c in "\r\n"):
        raise RealCallDenied("Real application credential is missing or invalid")
    return url, key


def application_policy(path: Path, *, real: bool) -> Policy:
    policy = load_policy(path)
    if os.environ.get("RELAY_PROVIDER", policy.provider) == "groq":
        if not real:
            raise RealCallDenied("Groq application requires --real")
        return Policy.model_validate(policy.model_dump() | {"provider": "groq"})
    return policy
