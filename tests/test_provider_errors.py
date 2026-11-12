from datetime import UTC, datetime

import pytest

from relay.obs.clock import FakeClock
from relay.obs.events import Sanitizer
from relay.providers.errors import classify_error, header_snapshot, retry_after_ms
from relay.providers.settings import ErrorRule, provider_settings


def fixture_rules() -> list[ErrorRule]:
    return [
        ErrorRule(kind=kind, field_path="error.code", pattern=pattern, source="fixture")
        for kind, pattern in [
            ("rate_limit_minute", "^FAKE_MINUTE$"),
            ("rate_limit_day", "^FAKE_DAY$"),
        ]
    ]


@pytest.mark.parametrize(
    "body,kind",
    [
        ('{"error":{"code":"FAKE_MINUTE"}}', "rate_limit_minute"),
        ('{"error":{"code":"FAKE_DAY"}}', "rate_limit_day"),
        ('{"error":{"message":"unrecognised"}}', "rate_limit_unknown"),
        ("not JSON", "rate_limit_unknown"),
    ],
)
def test_fixture_classification_and_retry_after(body: str, kind: str) -> None:
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC))
    safe = Sanitizer()
    snapshot = header_snapshot({"Retry-After": "2"}, clock=clock, sanitizer=safe)
    error = classify_error(429, body, snapshot, rules=fixture_rules(), sanitizer=safe)
    assert error.kind == kind and error.retry_after_ms == 2000
    assert error.status_code == 429


def test_request_headers_are_rpd_and_tokens_are_tpm() -> None:
    clock = FakeClock(utc=datetime(2026, 11, 12, tzinfo=UTC), monotonic_ns=123)
    snapshot = header_snapshot(
        {
            "X-RateLimit-Limit-Requests": "1000",
            "X-RateLimit-Remaining-Requests": "999",
            "X-RateLimit-Limit-Tokens": "8000",
            "X-RateLimit-Remaining-Tokens": "7000",
            "X-RateLimit-Reset-Tokens": "7.66s",
            "X-RateLimit-Future-Input": "unknown meaning",
            "Authorization": "Bearer FAKE-canary",
            "set-cookie": "not safe",
        },
        clock=clock,
        sanitizer=Sanitizer(),
    )
    assert snapshot.limit_requests_rpd == 1000 and snapshot.remaining_requests_rpd == 999
    assert snapshot.limit_tokens_tpm == 8000 and snapshot.remaining_tokens_tpm == 7000
    assert snapshot.observed_ns == 123
    assert snapshot.headers["x-ratelimit-future-input"] == "unknown meaning"
    assert "authorization" not in snapshot.headers and "set-cookie" not in snapshot.headers
    assert not hasattr(snapshot, "rpm") and not hasattr(snapshot, "tpd")


def test_default_rules_do_not_invent_real_error_shapes() -> None:
    assert provider_settings().error_rules == []
    for invalid in (None, "tomorrow", "nan", "inf", "-2"):
        assert retry_after_ms(invalid) is None
    assert retry_after_ms("0.25") == 250
