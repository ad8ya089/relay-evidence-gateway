"""The sole structured-event writer, with recursive credential redaction."""

import json
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr

from relay.obs.clock import Clock, SystemClock

MAX_BODY_CHARS = 500  # Security requirement, not a calibrated policy value.
REDACTED = "[REDACTED]"


class EventWriteError(RuntimeError):
    """Invalid event or failed persistence; never silently drop an event."""


class Sanitizer:
    def __init__(self, secrets: Sequence[str] = ()) -> None:
        self._secrets = set(secret for secret in secrets if secret)

    def register_secret(self, secret: str) -> None:
        if secret:
            self._secrets.add(secret)

    @staticmethod
    def sensitive_key(key: str) -> bool:
        normal = re.sub(r"[^a-z0-9]", "", key.casefold())
        return normal in {"env", "environ", "environment", "token"} or any(
            term in normal
            for term in (
                "authorization",
                "apikey",
                "password",
                "secret",
                "credential",
                "accesstoken",
                "refreshtoken",
            )
        )

    def text(self, value: str, *, body: bool = False) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            value = value.replace(secret, REDACTED)
        value = re.sub(r"gsk_[A-Za-z0-9_-]+", REDACTED, value)
        value = re.sub(r"(?i)\bBearer\s+[^\s,\"'}]+", REDACTED, value)
        value = re.sub(
            r"(?i)\b(?:proxy-)?authorization\b[\"']?\s*[:=]\s*[^\r\n,}]+",
            REDACTED,
            value,
        )
        return value[:MAX_BODY_CHARS] if body else value

    def value(self, value: object, *, key: str = "") -> JsonValue:
        if isinstance(value, SecretStr) or self.sensitive_key(key):
            return REDACTED
        if type(value).__name__ == "_Environ":
            return REDACTED
        if isinstance(value, str):
            return self.text(value, body="body" in key.casefold())
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, BaseModel):
            return self.value(value.model_dump(mode="json"), key=key)
        if isinstance(value, Mapping):
            result: dict[str, JsonValue] = {}
            for name, item in value.items():
                if not isinstance(name, str):
                    raise EventWriteError("Event dictionary keys must be strings")
                if "authorization" in re.sub(r"[^a-z]", "", name.casefold()):
                    continue  # Do not persist even the header name.
                result[self.text(name)] = self.value(item, key=name)
            return result
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        raise EventWriteError("Event contains an unsupported non-JSON value")


class EventContext(BaseModel):
    rid: str
    run: str
    arrival_ns: int = Field(ge=0)


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    rid: str
    run: str
    t_ms: float = Field(ge=0)
    ev: str


class EventWriter:
    def __init__(
        self, path: Path, *, clock: Clock | None = None, secrets: Sequence[str] = ()
    ) -> None:
        self.path = path
        self.clock = clock if clock is not None else SystemClock()
        self.sanitizer = Sanitizer(secrets)
        self._lock = threading.RLock()

    def write(self, context: EventContext, ev: str, **fields: object) -> Event:
        if {"rid", "run", "t_ms", "ev"}.intersection(fields):
            raise EventWriteError("Event fields cannot override their request context")
        try:
            with self._lock:
                payload = self.sanitizer.value(
                    {
                        "rid": context.rid,
                        "run": context.run,
                        "t_ms": (self.clock.monotonic_ns() - context.arrival_ns) / 1_000_000,
                        "ev": ev,
                        **fields,
                    }
                )
                event = Event.model_validate(payload)
                line = json.dumps(event.model_dump(mode="json"), allow_nan=False)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                return event
        except (OSError, ValueError, TypeError) as exc:
            raise EventWriteError(f"Cannot persist event ({type(exc).__name__})") from None

    async def emit(self, context: EventContext, ev: str, **fields: object) -> Event:
        # Short synchronous append is protected across threads as well as async callers.
        return self.write(context, ev, **fields)
