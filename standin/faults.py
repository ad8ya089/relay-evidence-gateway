"""Dedicated seeded sampling, with public safe decisions for reproducibility."""

import random
import threading
from typing import Literal

from pydantic import BaseModel

from standin.distributions import KindFit, draw_pair
from standin.profile import Distribution, Profile


class Decision(BaseModel):
    sequence: int
    seed: int
    kind: str
    inflight: int
    fault: Literal["none", "server", "timeout", "disconnect"]
    slow_success: bool
    verify_fail: bool
    ttft_ms: float
    decode_tps: float
    completion_tokens: int
    load_multiplier: float
    latency_multiplier: float

    def decode_ms(self, actual_completion_tokens: int) -> float:
        return actual_completion_tokens / self.decode_tps * 1000 * self.latency_multiplier


def sample(distribution: Distribution, rng: random.Random) -> float:
    values = distribution.parameters.value
    if distribution.family == "lognormal":
        return rng.lognormvariate(*values)
    if distribution.family == "empirical":
        return rng.choice(values)
    return values[0]


class Sampler:
    def __init__(
        self,
        profile: Profile,
        rng: random.Random | None = None,
        calibration: dict[str, KindFit] | None = None,
    ) -> None:
        self.rng = rng if rng is not None else random.Random(profile.seed.value)
        self._initial_state = self.rng.getstate()
        self._sequence = 0
        self._lock = threading.RLock()
        self.calibration = calibration or {}

    def reset(self) -> None:
        with self._lock:
            self.rng.setstate(self._initial_state)
            self._sequence = 0

    def decide(
        self,
        profile: Profile,
        kind: str,
        inflight: int,
        maximum: int,
        *,
        use_calibration: bool = True,
    ) -> Decision:
        with self._lock:
            self._sequence += 1
            faults = profile.faults
            # Independent draws, deterministic precedence for mutually exclusive failures.
            server, timeout, disconnect, slow, verify = [self.rng.random() for _ in range(5)]
            fault = "none"
            if server < faults.server_error_prob.value:
                fault = "server"
            elif timeout < faults.timeout_prob.value:
                fault = "timeout"
            elif disconnect < faults.disconnect_prob.value:
                fault = "disconnect"
            slow_success = fault == "none" and slow < faults.slow_success_prob.value
            multiplier = 1 + profile.alpha.value * max(
                0, (inflight - profile.capacity.value) / profile.capacity.value
            )
            latency = multiplier * (faults.slow_factor.value if slow_success else 1)
            timing = profile.timing[kind]
            fitted = self.calibration.get(kind) if use_calibration else None
            ttft, rate, tokens = (
                draw_pair(fitted, self.rng) if fitted is not None else (None, None, None)
            )
            return Decision(
                sequence=self._sequence,
                seed=profile.seed.value,
                kind=kind,
                inflight=inflight,
                fault=fault,
                slow_success=slow_success,
                verify_fail=verify < faults.verify_fail_prob.value,
                ttft_ms=(ttft if ttft is not None else sample(timing.ttft_ms, self.rng)) * latency,
                decode_tps=rate if rate is not None else sample(timing.decode_tps, self.rng),
                completion_tokens=min(
                    maximum,
                    max(
                        1,
                        round(
                            tokens
                            if tokens is not None
                            else sample(timing.completion_tokens, self.rng)
                        ),
                    ),
                ),
                load_multiplier=multiplier,
                latency_multiplier=latency,
            )
