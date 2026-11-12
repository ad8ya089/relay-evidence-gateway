import math
import random
from statistics import NormalDist

import pytest

from standin.calibrate import Sample, calibrate, estimate_alpha, extract, fit_kind, settings
from standin.distributions import draw_pair
from standin.profile import Profile, load_profile
from standin.service import create_app
from tests.standin_helpers import clock, profile
from tests.test_standin_service import request


def fixture_events(kind: str, n: int, mu: float, sigma: float) -> list[dict[str, object]]:
    rows = []
    for i in range(n):
        z = NormalDist().inv_cdf((i + 0.5) / n)
        ttft, rate = math.exp(mu + sigma * z), math.exp(5 + 0.2 * z)
        common = {
            "rid": f"FAKE-r{i}",
            "run": "FAKE-fixture",
            "call_id": f"{kind}-{i}",
            "synthetic": True,
        }
        rows.extend(
            [
                common
                | {
                    "ev": "call_start",
                    "source": "real",
                    "kind": kind,
                    "model": profile().model,
                    "t_ms": 10,
                },
                common | {"ev": "first_delta", "what": "reasoning", "provider_elapsed_ms": 1},
                common | {"ev": "first_delta", "what": "content", "provider_elapsed_ms": ttft},
                common
                | {
                    "ev": "call_end",
                    "source": "real",
                    "status": 200,
                    "total_ms": ttft + 100 / rate * 1000,
                    "usage": {
                        "in": 30,
                        "cached": 10,
                        "out": 100,
                        "total_tokens": 130,
                        "completion_time": 100 / rate,
                    },
                },
            ]
        )
    return rows


def test_synthetic_event_fixture_recovers_separate_lognormals_within_ten_percent() -> None:
    events = fixture_events("draft", 1000, 5, 0.3) + fixture_events("verify", 1000, 4, 0.5)
    # Explicit in-memory fixture-only extractor; production CLI has no bypass flag.
    samples = extract(events, profile().model, fixture_mode=True).samples
    for kind, mu, sigma in (("draft", 5, 0.3), ("verify", 4, 0.5)):
        fit = fit_kind([s for s in samples if s.kind == kind], settings(), random.Random(1))
        assert fit.ttft_ms.mu == pytest.approx(mu, rel=0.1)
        assert fit.ttft_ms.sigma == pytest.approx(sigma, rel=0.1)
        assert fit.decode_tps.mu == pytest.approx(5, rel=0.1)
        assert fit.decode_tps.sigma == pytest.approx(0.2, rel=0.1)
        assert fit.method == "lognormal_percentiles"
        assert fit.decode_methods == {"provider_completion_time": 1000}
        assert fit.completion_token_samples == [100] * 1000


def test_source_selection_never_fits_synthetic_or_standin_output() -> None:
    events = fixture_events("draft", 5, 5, 0.3)
    events += [
        dict(row, source="stand-in", synthetic=False, run="local-run", rid="r", call_id="c")
        for row in events[:4]
    ]
    report = calibrate(events, settings(), load_profile(), clock=clock())
    assert report.status == "uncalibrated" and report.n_calls == {"draft": 0, "verify": 0}
    assert report.kinds["draft"].ttft_ms is None
    assert report.kinds["draft"].method == "unavailable"
    assert report.alpha.source == "assumed" and report.alpha.value == 0.5
    assert report.faults["server_error_prob"].source == "chosen"
    assert report.faults["server_error_prob"].value == 0.025


def test_small_sample_bootstrap_preserves_observed_pairs_and_warns_p99() -> None:
    samples = extract(
        fixture_events("draft", 10, 5, 0.3), profile().model, fixture_mode=True
    ).samples
    config = settings().model_copy(
        update={"tail_probability": settings().tail_probability.model_copy(update={"value": 0})}
    )
    fit = fit_kind(samples, config, random.Random(7))
    assert fit.method == "paired_bootstrap_lognormal_tail"
    rng = random.Random(4)
    assert all(draw_pair(fit, rng) in fit.pairs for _ in range(100))
    assert any("p99 is unreliable" in w for w in fit.warnings)
    tail = fit.model_copy(update={"tail_probability": 1})
    ttft, rate, _ = draw_pair(tail, random.Random(10))
    assert ttft >= max(p[0] for p in fit.pairs)
    assert rate >= max(p[1] for p in fit.pairs)


def test_nonstream_ttft_absent_and_decode_fallback_is_visible() -> None:
    rows = fixture_events("draft", 1, 5, 0.3)
    rows[-1]["usage"].pop("completion_time")
    sample = extract(rows, profile().model, fixture_mode=True).samples[0]
    assert sample.decode_method == "first_content_to_completion"
    no_deltas = [e for e in rows if e["ev"] != "first_delta"]
    sample = extract(no_deltas, profile().model, fixture_mode=True).samples[0]
    assert sample.ttft_ms is None and sample.decode_tps is None
    no_deltas[-1]["usage"]["completion_time"] = 0.5
    sample = extract(no_deltas, profile().model, fixture_mode=True).samples[0]
    assert sample.ttft_ms is None and sample.decode_tps == 200
    rows[-1]["usage"].pop("cached")
    assert extract(rows, profile().model, fixture_mode=True).samples[0].decode_tps is None


def test_alpha_requires_explicit_comparable_burst_six() -> None:
    def sample(run: str, total: float) -> Sample:
        return Sample(
            call_id="c",
            run=run,
            model="m",
            kind="draft",
            total_ms=total,
            ttft_ms=None,
            decode_tps=None,
            decode_method=None,
            completion_tokens=100,
        )

    samples = [sample("seq", 100)] * 10 + [sample("burst", 200)] * 6
    config = settings().model_copy(
        update={"sequential_run_ids": ["seq"], "burst_run_ids": ["burst"]}
    )
    assert estimate_alpha(samples, config, 2).source == "fitted"
    assert estimate_alpha(samples, config, 2).value == 0.5
    assert estimate_alpha(samples, config, 6).source == "assumed"
    assert estimate_alpha(samples[:-1], config, 2).source == "assumed"


def test_fitted_runtime_sampling_is_explicit_and_fallback_yaml_is_unchanged() -> None:
    samples = extract(
        fixture_events("generic", 5, 5, 0.3), profile().model, fixture_mode=True
    ).samples
    # Generic is excluded; use draft samples for an explicit in-memory test overlay.
    assert not samples
    samples = extract(
        fixture_events("draft", 5, 5, 0.3), profile().model, fixture_mode=True
    ).samples
    fit = fit_kind(samples, settings(), random.Random(1))
    original = load_profile().model_dump()
    report = calibrate([], settings(), profile(), clock=clock()).model_copy(
        update={"status": "calibrated", "kinds": {"generic": fit}}
    )
    app = create_app(profile=profile(), clock=clock(), calibration=report)
    prepared = app.state.simulation.begin(request())
    assert prepared.decision.ttft_ms > 0
    assert app.state.simulation.sampler.calibration["generic"] == fit
    app.state.simulation.finish(prepared, success=True)
    assert load_profile().model_dump() == original
    data = profile().model_dump()
    data["models"]["second-model"] = data["models"][data["model"]]
    timing = data["timing"]["generic"]["ttft_ms"]
    timing["family"], timing["parameters"]["value"] = "constant", [10000]
    app = create_app(profile=Profile.model_validate(data), clock=clock(), calibration=report)
    other = app.state.simulation.begin(request().model_copy(update={"model": "second-model"}))
    assert other.decision.ttft_ms == 10000
    app.state.simulation.finish(other, success=True)


def test_fixture_source_selection_keeps_only_real_labelled_calls() -> None:
    real_shape = fixture_events("draft", 2, 5, 0.3)
    local = [dict(row, source="stand-in", run="local") for row in real_shape]
    extraction = extract(real_shape + local, profile().model, fixture_mode=True)
    assert len(extraction.samples) == 2
    assert extraction.excluded["non_real_calls"] == 2
    collision = dict(real_shape[2], source="stand-in", provider_elapsed_ms=0.01)
    clean = extract(real_shape, profile().model, fixture_mode=True)
    mixed = extract(real_shape + [collision], profile().model, fixture_mode=True)
    assert [s.ttft_ms for s in clean.samples] == [s.ttft_ms for s in mixed.samples]


def test_large_p99_mismatch_is_reported() -> None:
    samples = [
        Sample(
            call_id=str(i),
            run="FAKE",
            model="FAKE",
            kind="draft",
            total_ms=100,
            ttft_ms=100,
            decode_tps=100,
            decode_method="provider_completion_time",
            completion_tokens=100,
        )
        for i in range(10)
    ]
    fit = fit_kind(samples, settings(), random.Random(1))
    assert fit.ratios["p99"] > 1.25
    assert any("more than 25%" in w for w in fit.warnings)
