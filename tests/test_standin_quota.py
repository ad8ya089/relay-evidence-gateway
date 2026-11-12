from concurrent.futures import ThreadPoolExecutor

import pytest

from standin.profile import Limits
from standin.quota import Quotas
from tests.standin_helpers import clock, profile


def limits(**values: int) -> Limits:
    p = profile()
    data = p.models[p.model].model_dump()
    for name, value in values.items():
        data[name]["value"] = value
    return Limits.model_validate(data)


@pytest.mark.parametrize("scope", ["rpm", "tpm", "rpd", "tpd"])
def test_each_scope_blocks_before_over_admission(scope: str) -> None:
    c = clock()
    q = Quotas(c)
    caps = limits(**{scope: 1 if scope in {"rpm", "rpd"} else 10})
    first = q.reserve("model", caps, 10)
    q.settle("model", first.charge_id, 10)
    rejected = q.reserve("model", caps, 10)
    assert rejected.scopes == [scope]
    assert rejected.retry_after_s == (60 if scope in {"rpm", "tpm"} else 30)
    assert rejected.view.headers["x-ratelimit-limit-requests"] == str(caps.rpd.value)
    assert rejected.view.headers["x-ratelimit-limit-tokens"] == str(caps.tpm.value)


def test_rolling_and_utc_reset_and_multiple_models() -> None:
    c, caps = clock(), limits(rpm=1, rpd=1)
    q = Quotas(c)
    accepted = q.reserve("a", caps, 100)
    q.settle("a", accepted.charge_id, 50)
    assert q.reserve("b", caps, 100).accepted
    c.advance_ms(30000)
    assert q.view("a", caps).usage["rpd"] == 0
    assert q.view("a", caps).usage["rpm"] == 1
    c.advance_ms(30000)
    assert q.reserve("a", caps, 100).accepted


def test_simultaneous_limits_use_longest_reset() -> None:
    c = clock()
    c.advance_ms(30000)
    caps = limits(rpm=1, tpm=10, rpd=1, tpd=10)
    q = Quotas(c)
    first = q.reserve("a", caps, 10)
    q.settle("a", first.charge_id, 10)
    result = q.reserve("a", caps, 10)
    assert set(result.scopes) == {"rpm", "tpm", "rpd", "tpd"}
    assert result.retry_after_s == 86400


def test_parallel_reservations_and_reconciliation() -> None:
    q, caps = Quotas(clock()), limits(tpm=100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: q.reserve("a", caps, 60), range(8)))
    admitted = [r for r in results if r.accepted]
    assert len(admitted) == 1
    assert all(r.retry_after_s is None for r in results if not r.accepted)
    q.settle("a", admitted[0].charge_id, 20)
    assert q.reserve("a", caps, 60).accepted


def test_old_pending_reservations_stay_charged() -> None:
    c = clock()
    q, caps = Quotas(c), limits(tpm=100, tpd=100)
    q.reserve("a", caps, 90)
    c.advance_ms(90000)
    assert q.view("a", caps).usage["tpm"] == q.view("a", caps).usage["tpd"] == 90
    with pytest.raises(ValueError, match="outstanding"):
        q.reset()
