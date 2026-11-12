"""Conservative, configurable HTTP classification and safe header observations."""

import json
import math
import re
from collections.abc import Mapping

from relay.obs.clock import Clock
from relay.obs.events import Sanitizer
from relay.providers.base import ErrorKind, HeaderSnapshot, ProviderFailure
from relay.providers.settings import ErrorRule


def retry_after_ms(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds * 1000 if math.isfinite(seconds) and seconds >= 0 else None


def _quota_int(value: str | None) -> int | None:
    try:
        number = int(value) if value is not None else -1
    except ValueError:
        return None
    return number if number >= 0 else None


def header_snapshot(
    headers: Mapping[str, str], *, clock: Clock, sanitizer: Sanitizer
) -> HeaderSnapshot:
    observed_ns = clock.monotonic_ns()
    safe = {
        key.casefold(): sanitizer.text(value, body=True)
        for key, value in headers.items()
        if (
            key.casefold().startswith("x-ratelimit-")
            or key.casefold() in {"retry-after", "x-request-id", "request-id", "x-groq-id"}
        )
        and not sanitizer.sensitive_key(key)
    }
    return HeaderSnapshot(
        observed_ns=observed_ns,
        headers=safe,
        limit_requests_rpd=_quota_int(safe.get("x-ratelimit-limit-requests")),
        remaining_requests_rpd=_quota_int(safe.get("x-ratelimit-remaining-requests")),
        limit_tokens_tpm=_quota_int(safe.get("x-ratelimit-limit-tokens")),
        remaining_tokens_tpm=_quota_int(safe.get("x-ratelimit-remaining-tokens")),
        retry_after_ms=retry_after_ms(safe.get("retry-after")),
    )


def _field(body: object, path: str) -> object:
    for name in path.split("."):
        if not isinstance(body, dict):
            return None
        body = body.get(name)
    return body


def classify_error(
    status: int,
    body: str,
    snapshot: HeaderSnapshot,
    *,
    rules: list[ErrorRule],
    sanitizer: Sanitizer,
) -> ProviderFailure:
    kind: ErrorKind
    context: dict[str, str] = {}
    if status == 429:
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            parsed = None
        matches = {
            rule.kind
            for rule in rules
            if isinstance(value := _field(parsed, rule.field_path), str)
            and re.search(rule.pattern, value, re.I)
        }
        kind = next(iter(matches)) if len(matches) == 1 else "rate_limit_unknown"
        if kind == "rate_limit_unknown":
            context["classification"] = (
                "No unique configured scope evidence; Retry-After is not scope"
            )
    elif status in {401, 403}:
        kind = "auth"
    elif status == 408:
        kind = "timeout"
    elif status >= 500:
        kind = "server"
    else:
        kind = "bad_request"
    return ProviderFailure(
        kind=kind,
        status_code=status,
        retry_after_ms=snapshot.retry_after_ms,
        body_trunc=sanitizer.text(body, body=True),
        headers=snapshot.headers,
        safe_context=context,
    )
