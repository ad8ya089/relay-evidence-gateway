"""Atomic simulation lifecycle, separate from HTTP and wire formatting."""

import random
import threading
from collections import Counter, deque

from pydantic import BaseModel, ConfigDict

from relay.obs.clock import Clock
from relay.providers.base import LLMRequest
from relay.retrieval.tokenizer import get_encoding
from standin.cache import CacheLookup, PrefixCache, prompt_tokens
from standin.distributions import KindFit
from standin.faults import Decision, Sampler
from standin.generation import InvalidInput, call_kind, generate
from standin.profile import Profile
from standin.quota import Admission, Quotas


class Prepared(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    profile: Profile
    request: LLMRequest
    decision: Decision
    cache: CacheLookup
    admission: Admission
    model: str
    id: str
    created: int
    text: str
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    reasoning_tokens: int

    @property
    def counted_tokens(self) -> int:
        return self.tokens_in - self.tokens_cached + self.tokens_out

    @property
    def usage(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.tokens_in,
            "completion_tokens": self.tokens_out,
            "total_tokens": self.tokens_in + self.tokens_out,
            "prompt_tokens_details": {"cached_tokens": self.tokens_cached},
            "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
            "queue_time": 0.0,
            "prompt_time": self.decision.ttft_ms / 1000,
            "completion_time": self.decision.decode_ms(self.tokens_out) / 1000,
        }


class QuotaRejected(Exception):
    def __init__(self, admission: Admission) -> None:
        self.admission = admission
        super().__init__("Simulated quota exhausted")


class Simulation:
    def __init__(
        self,
        profile: Profile,
        clock: Clock,
        rng: random.Random | None = None,
        calibration: dict[str, KindFit] | None = None,
    ) -> None:
        self.profile = profile.runnable()
        self.clock = clock
        self.lock = threading.RLock()
        self.sampler = Sampler(profile, rng, calibration)
        self.quotas = Quotas(clock, profile.window_s.value)
        self.cache = PrefixCache(clock)
        self.inflight = 0
        self.active: set[int] = set()
        self.stats: Counter[str] = Counter()
        self.fault_counts: Counter[str] = Counter()
        self.decisions: deque[Decision] = deque(maxlen=profile.diagnostic_history.value)
        self.last_admin_update: dict[str, object] | None = None

    def begin(self, request: LLMRequest) -> Prepared:
        with self.lock:
            profile = self.profile
            model = request.model or profile.model
            if model not in profile.models:
                raise InvalidInput("Model is not configured in this stand-in profile")
            kind = call_kind(request.messages)
            decision = self.sampler.decide(
                profile,
                kind,
                self.inflight + 1,
                request.max_completion_tokens,
                use_calibration=model == profile.model,
            )
            text = generate(request.messages, verify_fail=decision.verify_fail)
            visible_tokens = len(get_encoding().encode(text))
            if visible_tokens > request.max_completion_tokens:
                raise InvalidInput(
                    "Completion cap is too small for the simulator's complete JSON output"
                )
            tokens_out = max(visible_tokens, decision.completion_tokens)
            lookup = self.cache.lookup(model, request.messages, profile.cache)
            tokens_in = prompt_tokens(request.messages)
            cached = min(tokens_in, lookup.cached_tokens)
            admission = self.quotas.reserve(
                model, profile.models[model], tokens_in - cached + request.max_completion_tokens
            )
            if not admission.accepted:
                self.stats["rejected"] += 1
                raise QuotaRejected(admission)
            assert admission.charge_id is not None
            self.active.add(admission.charge_id)
            self.inflight += 1
            self.stats["accepted"] += 1
            self.fault_counts[decision.fault] += 1
            if decision.slow_success:
                self.fault_counts["slow_success"] += 1
            if kind == "verify" and decision.verify_fail:
                self.fault_counts["verify_fail"] += 1
            self.decisions.append(decision)
            self.cache.record_accepted(lookup)
            return Prepared(
                profile=profile,
                request=request,
                decision=decision,
                cache=lookup,
                admission=admission,
                model=model,
                id=f"sim-{profile.seed.value}-{admission.charge_id}",
                created=int(self.clock.utc_now().timestamp()),
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_cached=cached,
                reasoning_tokens=min(
                    tokens_out - visible_tokens, int(tokens_out * profile.reasoning_fraction.value)
                ),
            )

    def finish(self, prepared: Prepared, *, success: bool) -> None:
        with self.lock:
            charge_id = prepared.admission.charge_id
            if charge_id not in self.active:
                return  # Stream finalizer/background cleanup can both run.
            assert charge_id is not None
            self.quotas.settle(
                prepared.model, charge_id, prepared.counted_tokens if success else None
            )
            if success:
                self.cache.populate(prepared.cache, prepared.profile.cache)
            self.active.remove(charge_id)
            self.inflight -= 1
            self.stats["completed" if success else "failed_or_cancelled"] += 1

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "source": "stand-in",
                "profile": self.profile.model_dump(mode="json"),
                "model": self.profile.model,
                "seed": self.profile.seed.value,
                "calibration_kinds": list(self.sampler.calibration),
                "calibration_model": self.profile.model if self.sampler.calibration else None,
                "inflight": self.inflight,
                "capacity": self.profile.capacity.value,
                "models": {
                    m: self.quotas.view(m, limits).model_dump()
                    for m, limits in self.profile.models.items()
                },
                "statistics": dict(self.stats),
                "fault_counts": dict(self.fault_counts),
                "cache": self.cache.state(),
                "last_admin_update": self.last_admin_update,
                "decisions": [d.model_dump() for d in self.decisions],
            }

    def response_headers(self, model: str) -> dict[str, str]:
        with self.lock:
            return self.quotas.view(model, self.profile.models[model]).headers
