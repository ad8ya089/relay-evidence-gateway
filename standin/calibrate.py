"""Fit genuine P2 events only. No observations means explicitly uncalibrated."""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from relay.obs.clock import Clock, SystemClock
from relay.obs.events import Sanitizer
from relay.providers.base import Usage
from standin.distributions import KindFit, draw_pair, fit_metric, percentiles
from standin.profile import Parameter, Profile, load_profile


class CalibrationSettings(BaseModel):
    events: Path
    output: Path
    seed: Parameter[int]
    bootstrap_threshold: Parameter[int]
    p99_min_samples: Parameter[int]
    simulation_draws: Parameter[int]
    tail_probability: Parameter[float]
    p99_ratio_tolerance: Parameter[float]
    sequential_run_ids: list[str]
    burst_run_ids: list[str]

    @model_validator(mode="after")
    def valid_settings(self) -> "CalibrationSettings":
        if (
            min(
                self.bootstrap_threshold.value,
                self.p99_min_samples.value,
                self.simulation_draws.value,
            )
            <= 0
        ):
            raise ValueError("Calibration sample settings must be positive")
        if (
            not 0 <= self.tail_probability.value < 1
            or not math.isfinite(self.p99_ratio_tolerance.value)
            or self.p99_ratio_tolerance.value < 0
        ):
            raise ValueError("Invalid calibration tail probability or percentile tolerance")
        return self


class Sample(BaseModel):
    call_id: str
    run: str
    model: str
    kind: str
    total_ms: float
    ttft_ms: float | None
    decode_tps: float | None
    decode_method: str | None
    completion_tokens: int | None


class Extraction(BaseModel):
    samples: list[Sample] = Field(default_factory=list)
    attempts: int = 0
    errors: dict[str, int] = Field(default_factory=dict)
    excluded: dict[str, int] = Field(default_factory=dict)
    run_ids: list[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    status: Literal["calibrated", "uncalibrated"]
    generated_at: str
    model: str
    seed: int
    source_events: list[str]
    run_ids: list[str]
    n_calls: dict[str, int]
    kinds: dict[str, KindFit]
    alpha: Parameter[float]
    faults: dict[str, Parameter[float]]
    attempts: int
    excluded: dict[str, int]
    methods: list[str]
    limitations: list[str]


@lru_cache(maxsize=1)
def settings() -> CalibrationSettings:
    return CalibrationSettings.model_validate(
        yaml.safe_load(Path("config/calibration.yaml").read_text())
    )


def number(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        return float(value)
    return None


def synthetic(event: dict[str, object]) -> bool:
    if any(event.get(name) for name in ("synthetic", "fixture", "test", "unit_test")):
        return True
    text = " ".join(str(event.get(name, "")) for name in ("run", "model", "rid", "call_id"))
    return bool(
        re.search(r"(?i)(fake|fixture|unit.test|synthetic|(^|[-_ ])test([-_ ]|$)|unassigned)", text)
    )


def extract(
    events: list[dict[str, object]], model: str, *, fixture_mode: bool = False
) -> Extraction:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    for event in events:
        if not fixture_mode and synthetic(event):
            excluded["synthetic_events"] += 1
            continue
        if isinstance(event.get("call_id"), str) and isinstance(event.get("run"), str):
            grouped[(event["run"], event["call_id"])].append(event)
    result = Extraction()
    errors: Counter[str] = Counter()
    runs = set()
    for (run, call_id), rows in grouped.items():
        starts = [e for e in rows if e.get("ev") == "call_start"]
        ends = [e for e in rows if e.get("ev") in {"call_end", "call_error"}]
        if len(starts) != 1 or len(ends) != 1:
            excluded["incomplete_or_duplicate_calls"] += 1
            continue
        start, end = starts[0], ends[0]
        if start.get("source") != "real" or end.get("source") != "real":
            excluded["non_real_calls"] += 1
            continue
        if start.get("model") != model or end.get("model", model) != model:
            excluded["other_models"] += 1
            continue
        if start.get("kind") not in {"draft", "verify"} or start.get("rid") != end.get("rid"):
            excluded["invalid_kind_or_identity"] += 1
            continue
        result.attempts += 1
        runs.add(run)
        status = end.get("status")
        total = number(end.get("total_ms"))
        if (
            end.get("ev") != "call_end"
            or not isinstance(status, int)
            or not 200 <= status < 300
            or total is None
        ):
            errors[str(end.get("reason", "unknown"))] += 1
            continue
        deltas = [
            e
            for e in rows
            if e.get("ev") == "first_delta"
            and e.get("what") == "content"
            and e.get("rid") == start.get("rid")
        ]
        times = [number(e.get("provider_elapsed_ms")) for e in deltas]
        ttft = min((t for t in times if t is not None and t <= total), default=None)
        try:
            usage = Usage.model_validate(end.get("usage", {}))
        except ValueError:
            usage = Usage()
        tokens = usage.tokens_out if usage.complete and usage.tokens_out else None
        decode_time = number(usage.completion_time)
        method = "provider_completion_time" if decode_time is not None else None
        if decode_time is None and ttft is not None and total > ttft:
            decode_time, method = (total - ttft) / 1000, "first_content_to_completion"
        rate = tokens / decode_time if tokens and decode_time else None
        result.samples.append(
            Sample(
                call_id=call_id,
                run=run,
                model=model,
                kind=start["kind"],
                total_ms=total,
                ttft_ms=ttft,
                decode_tps=rate,
                decode_method=method if rate else None,
                completion_tokens=tokens,
            )
        )
    result.excluded, result.errors, result.run_ids = dict(excluded), dict(errors), sorted(runs)
    return result


def fit_kind(samples: list[Sample], config: CalibrationSettings, rng: random.Random) -> KindFit:
    ttfts = [s.ttft_ms for s in samples if s.ttft_ms is not None]
    rates = [s.decode_tps for s in samples if s.decode_tps is not None]
    pairs = [
        (s.ttft_ms, s.decode_tps, s.completion_tokens)
        for s in samples
        if s.ttft_ms is not None and s.decode_tps is not None and s.completion_tokens is not None
    ]
    small = len(pairs) < config.bootstrap_threshold.value
    fit = KindFit(
        n_calls=len(samples),
        method="paired_bootstrap_lognormal_tail" if small and pairs else "lognormal_percentiles",
        ttft_ms=fit_metric(ttfts),
        decode_tps=fit_metric(rates),
        pairs=pairs,
        completion_token_samples=[
            s.completion_tokens for s in samples if s.completion_tokens is not None
        ],
        decode_methods=dict(Counter(s.decode_method for s in samples if s.decode_method)),
        tail_probability=config.tail_probability.value,
    )
    # Compare only calls for which both observed latency components and usage are known.
    observed = [
        s.total_ms
        for s in samples
        if s.ttft_ms is not None and s.decode_tps is not None and s.completion_tokens is not None
    ]
    if observed and fit.ttft_ms and fit.decode_tps:
        simulated = []
        for _ in range(config.simulation_draws.value):
            ttft, rate, tokens = draw_pair(fit, rng)
            assert ttft is not None and rate is not None and tokens is not None
            simulated.append(ttft + tokens / rate * 1000)
        fit.observed_ms, fit.simulated_ms = percentiles(observed), percentiles(simulated)
        fit.ratios = {p: fit.simulated_ms[p] / v for p, v in fit.observed_ms.items()}
        if abs(fit.ratios["p99"] - 1) > config.p99_ratio_tolerance.value:
            fit.warnings.append("Simulated p99 differs from observed by more than 25%")
    if len(observed) < config.p99_min_samples.value:
        fit.warnings.append(f"n={len(observed)}: p99 is unreliable below 100 observations")
    if small:
        fit.warnings.append(
            "Small sample; bootstrap pairs retained, tail mass is chosen, not measured"
        )
    return fit


def estimate_alpha(
    samples: list[Sample], config: CalibrationSettings, capacity: int
) -> Parameter[float]:
    sequential = [s.total_ms for s in samples if s.run in config.sequential_run_ids]
    burst = [s.total_ms for s in samples if s.run in config.burst_run_ids]
    seq_kinds = {s.kind for s in samples if s.run in config.sequential_run_ids}
    burst_kinds = {s.kind for s in samples if s.run in config.burst_run_ids}
    if (
        sequential
        and len(burst) == 6
        and capacity < 6
        and seq_kinds == burst_kinds
        and len(seq_kinds) == 1
    ):
        ratio = (sum(burst) / len(burst)) / (sum(sequential) / len(sequential))
        alpha = (ratio - 1) / ((6 - capacity) / capacity)
        if math.isfinite(alpha) and alpha > 0:
            return Parameter(
                value=alpha,
                source="fitted",
                note="Declared sequential versus burst-six runs; caller supplies real provenance",
            )
    return Parameter(
        value=0.5, source="assumed", note="No usable genuine sequential/burst-six comparison"
    )


def calibrate(
    events: list[dict[str, object]],
    config: CalibrationSettings,
    profile: Profile,
    *,
    clock: Clock | None = None,
    source_events: list[str] | None = None,
) -> CalibrationReport:
    extraction = extract(events, profile.model)
    rng = random.Random(config.seed.value)
    kinds = {
        kind: fit_kind([s for s in extraction.samples if s.kind == kind], config, rng)
        for kind in ("draft", "verify")
    }
    faults = {
        name: getattr(profile.faults, name)
        for name in (
            "server_error_prob",
            "timeout_prob",
            "disconnect_prob",
            "slow_success_prob",
            "verify_fail_prob",
        )
    }
    for name, reason in (
        ("server_error_prob", "server"),
        ("timeout_prob", "timeout"),
        ("disconnect_prob", "disconnect"),
    ):
        count = extraction.errors.get(reason, 0)
        if extraction.attempts >= config.bootstrap_threshold.value and count:
            faults[name] = Parameter(
                value=count / extraction.attempts,
                source="fitted",
                note="Observed failure frequency; finite-sample estimate of underlying probability",
            )
    fitted = any(k.ttft_ms is not None or k.decode_tps is not None for k in kinds.values())
    return CalibrationReport(
        status="calibrated" if fitted else "uncalibrated",
        generated_at=(clock or SystemClock()).utc_now().isoformat(),
        model=profile.model,
        seed=config.seed.value,
        source_events=source_events or [],
        run_ids=extraction.run_ids,
        n_calls={kind: k.n_calls for kind, k in kinds.items()},
        kinds=kinds,
        alpha=estimate_alpha(extraction.samples, config, profile.capacity.value),
        faults=faults,
        attempts=extraction.attempts,
        excluded=extraction.excluded,
        methods=[
            "Join P2 events by run and call_id with matching request identity",
            "First actual content delta provider_elapsed_ms; reasoning excluded",
            "Prefer provider completion_time, otherwise total_ms minus first content time",
            "Empirical output tokens; paired bootstrap below threshold, conditional lognormal tail",
            "Compare paired complete samples only; no non-stream TTFT invented",
        ],
        limitations=[
            "Source labels cannot establish authenticity; review input event provenance",
            "Synthetic/test events excluded; genuine account data not supplied in P3",
            "No batching, cross-tenant noise or real enforcement-window fidelity",
            "Tail probability and unmeasured fault defaults remain chosen/assumed",
        ],
    )


def read_events(path: Path) -> tuple[list[dict[str, object]], int]:
    if not path.exists():
        return [], 0
    if any(part.casefold() in {"tests", "fixtures"} for part in path.parts):
        return [], 1
    rows, invalid = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Event must be an object")
            rows.append(value)
        except ValueError:
            invalid += 1
    return rows, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequential-run", action="append", default=[])
    parser.add_argument("--burst-run", action="append", default=[])
    args = parser.parse_args()
    try:
        config = settings()
        config = config.model_copy(
            update={
                "events": args.events or config.events,
                "output": args.output or config.output,
                "sequential_run_ids": args.sequential_run or config.sequential_run_ids,
                "burst_run_ids": args.burst_run or config.burst_run_ids,
            }
        )
        events, invalid = read_events(config.events)
        report = calibrate(events, config, load_profile(), source_events=[str(config.events)])
        report.excluded["invalid_lines_or_fixture_paths"] = invalid
        safe = Sanitizer().value(report)
        config.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = config.output.with_suffix(".json.part")
        temporary.write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(config.output)
        print(f"Calibration status: {report.status}; real samples: {report.n_calls}")
        for kind, fit in report.kinds.items():
            print(
                f"{kind}: observed={fit.observed_ms}; simulated={fit.simulated_ms}; "
                f"ratios={fit.ratios}"
            )
            for warning in fit.warnings:
                print(f"Warning: {warning}")
        return 0
    except (ValueError, OSError):
        print("Calibration refused: invalid configuration/events or unavailable output")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
