import pytest

from standin.faults import Sampler
from tests.standin_helpers import profile


def test_seed_reproducibility_and_reset() -> None:
    p = profile()
    sampler = Sampler(p)
    sequence = [sampler.decide(p, "draft", 1, 450) for _ in range(10)]
    sampler.reset()
    assert sequence == [sampler.decide(p, "draft", 1, 450) for _ in range(10)]
    changed = profile(seed=99)
    assert sequence != [Sampler(changed).decide(changed, "draft", 1, 450) for _ in range(10)]


def test_exact_slowdown_formula_and_slow_success() -> None:
    p = profile(capacity=2, alpha=0.5)
    one = Sampler(p).decide(p, "draft", 1, 450)
    six = Sampler(p).decide(p, "draft", 6, 450)
    assert six.load_multiplier == 2
    assert six.ttft_ms == pytest.approx(one.ttft_ms * 2)
    assert six.decode_ms(200) == pytest.approx(one.decode_ms(200) * 2)
    slow = profile(faults__slow_success_prob=1, faults__slow_factor=8)
    decision = Sampler(slow).decide(slow, "draft", 1, 450)
    assert decision.fault == "none" and decision.slow_success
    assert decision.ttft_ms == pytest.approx(one.ttft_ms * 8)


@pytest.mark.parametrize("fault", ["server_error", "timeout", "disconnect"])
def test_faults_have_distinct_decisions(fault: str) -> None:
    p = profile(**{"faults__" + fault + "_prob": 1})
    decision = Sampler(p).decide(p, "verify", 1, 120)
    assert decision.fault == {"server_error": "server"}.get(fault, fault)
    assert not decision.slow_success
