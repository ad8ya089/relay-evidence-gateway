"""Precomputed seeded arrival plans, independent of response completion."""

import math
import random
from decimal import ROUND_CEILING, Decimal
from typing import Literal

from pydantic import Field, model_validator

from loadgen.settings import Model


class Segment(Model):
    duration_s: float = Field(gt=0)
    rate_per_minute: float | None = Field(default=None, gt=0)
    multiplier: float | None = Field(default=None, gt=0)
    end_rate_per_minute: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def rate(self) -> "Segment":
        if (self.rate_per_minute is None) == (self.multiplier is None):
            raise ValueError("Supply one absolute rate or capacity multiplier")
        return self


class ArrivalConfig(Model):
    kind: Literal["constant", "step", "burst", "ramp", "one_shot"]
    segments: list[Segment] = Field(default_factory=list)
    count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def shape(self) -> "ArrivalConfig":
        if self.kind == "one_shot":
            if self.count is None or self.segments:
                raise ValueError("One-shot arrivals require count and no segments")
        elif not self.segments or self.count is not None:
            raise ValueError("Rate arrivals require segments and no count")
        if self.kind == "ramp" and any(s.end_rate_per_minute is None for s in self.segments):
            raise ValueError("Ramp segments need their ending rate")
        return self


class PlannedRequest(Model):
    request_id: str
    offset_ns: int = Field(ge=0)
    question_id: str
    sampling: Literal["uniform", "zipf"]
    sample_draw: float = Field(ge=0, lt=1)
    seed: int
    setting: Literal["A", "B", "C"]


def offsets(config: ArrivalConfig, maximum: int, capacity: float | None = None) -> list[int]:
    if config.kind == "one_shot":
        if config.count > maximum:
            raise ValueError("Arrival count exceeds configured task bound")
        return [0] * config.count
    result: list[int] = []
    elapsed = Decimal(0)
    for segment in config.segments:
        rate = segment.rate_per_minute
        if rate is None:
            if capacity is None or not math.isfinite(capacity) or capacity <= 0:
                raise ValueError("Measured positive capacity is required for multipliers")
            rate = segment.multiplier * capacity
        duration = Decimal(str(segment.duration_s))
        first = Decimal(str(rate)) / 60
        last = Decimal(str(segment.end_rate_per_minute or rate)) / 60
        count = int((duration * (first + last) / 2).to_integral_value(rounding=ROUND_CEILING))
        if len(result) + count > maximum:
            raise ValueError("Arrival count exceeds configured task bound")
        slope = (last - first) / duration
        for index in range(count):
            # Invert the integrated rate; never use a response to choose an arrival.
            t = (
                Decimal(index) / first
                if not slope
                else (first * first + 2 * slope * index).sqrt() - first
            )
            if slope:
                t /= slope
            if t < duration:
                result.append(int((elapsed + t) * 1_000_000_000))
        elapsed += duration
    return result


def plan(
    config: ArrivalConfig,
    question_ids: list[str],
    *,
    run_id: str,
    seed: int,
    setting: Literal["A", "B", "C"],
    sampling: Literal["uniform", "zipf"],
    maximum: int,
    exponent: float,
    capacity: float | None = None,
) -> list[PlannedRequest]:
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Unique nonempty question IDs are required")
    rng = random.Random(seed)
    weights = [
        1 / (i + 1) ** exponent if sampling == "zipf" else 1 for i in range(len(question_ids))
    ]
    total = sum(weights)
    result = []
    for index, offset in enumerate(offsets(config, maximum, capacity)):
        draw = rng.random()
        cumulative = 0.0
        selected = question_ids[-1]
        for qid, weight in zip(question_ids, weights, strict=True):
            cumulative += weight / total
            if draw < cumulative:
                selected = qid
                break
        result.append(
            PlannedRequest(
                request_id=f"{run_id}-{index}",
                offset_ns=offset,
                question_id=selected,
                sampling=sampling,
                sample_draw=draw,
                seed=seed,
                setting=setting,
            )
        )
    return result
