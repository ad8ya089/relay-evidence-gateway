"""Pure fitting and seeded paired-bootstrap sampling shared by calibration/runtime."""

import math
import random
from statistics import NormalDist
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricFit(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    source: Literal["fitted"] = "fitted"
    n: int = Field(gt=0)
    mu: float
    sigma: float = Field(ge=0)
    p50: float = Field(gt=0)
    p95: float = Field(gt=0)


class KindFit(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    n_calls: int = Field(ge=0)
    method: Literal["paired_bootstrap_lognormal_tail", "lognormal_percentiles"]
    ttft_ms: MetricFit | None
    decode_tps: MetricFit | None
    pairs: list[tuple[float, float, int]] = Field(default_factory=list)
    completion_token_samples: list[int] = Field(default_factory=list)
    decode_methods: dict[str, int] = Field(default_factory=dict)
    tail_probability: float = Field(ge=0, le=1)
    tail_source: Literal["chosen"] = "chosen"
    observed_ms: dict[str, float] = Field(default_factory=dict)
    simulated_ms: dict[str, float] = Field(default_factory=dict)
    ratios: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def positive_samples(self) -> "KindFit":
        if any(min(pair) <= 0 for pair in self.pairs) or any(
            t <= 0 for t in self.completion_token_samples
        ):
            raise ValueError("Calibration samples must be positive")
        return self


def percentiles(values: list[float]) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p, method="linear")) for p in (50, 95, 99)}


def fit_metric(values: list[float]) -> MetricFit | None:
    if not values:
        return None
    quantiles = percentiles(values)
    return MetricFit(
        n=len(values),
        mu=math.log(quantiles["p50"]),
        sigma=max(0, (math.log(quantiles["p95"]) - math.log(quantiles["p50"])) / 1.645),
        p50=quantiles["p50"],
        p95=quantiles["p95"],
    )


def tail_sample(fit: MetricFit, maximum: float, quantile: float) -> float:
    if fit.sigma == 0:
        return maximum
    normal = NormalDist(fit.mu, fit.sigma)
    floor = normal.cdf(math.log(maximum))
    probability = min(math.nextafter(1.0, 0.0), floor + (1 - floor) * quantile)
    return math.exp(normal.inv_cdf(probability))


def draw_pair(fit: KindFit, rng: random.Random) -> tuple[float | None, float | None, int | None]:
    if fit.method == "paired_bootstrap_lognormal_tail" and fit.pairs:
        ttft, decode, tokens = rng.choice(fit.pairs)
        if (
            rng.random() < fit.tail_probability
            and fit.ttft_ms is not None
            and fit.decode_tps is not None
        ):
            u = rng.random()
            ttft = tail_sample(fit.ttft_ms, max(p[0] for p in fit.pairs), u)
            decode = tail_sample(fit.decode_tps, max(p[1] for p in fit.pairs), u)
        return ttft, decode, tokens
    ttft = rng.lognormvariate(fit.ttft_ms.mu, fit.ttft_ms.sigma) if fit.ttft_ms else None
    decode = rng.lognormvariate(fit.decode_tps.mu, fit.decode_tps.sigma) if fit.decode_tps else None
    tokens = rng.choice(fit.completion_token_samples) if fit.completion_token_samples else None
    return ttft, decode, tokens
