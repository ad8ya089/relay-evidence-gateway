"""The single calculator for experimental tables, capacity and generator diagnostics."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from loadgen.settings import load_settings
from scripts.build_results_csv import json_rows, read_csv
from scripts.result_models import ResultRow

UNKNOWN = "not measured"
OFFICIAL = (
    "steady burst sustained_overload minute_limit_halved daily_exhausted "
    "standin_burst6_exact retry_naive retry_bounded burst_cache_on"
).split()


def percentiles(values: list[float], minimum: int = 2) -> dict[str, object]:
    result: dict[str, object] = {"n": len(values)}
    for p in (50, 95, 99):
        result[f"p{p}"] = (
            float(np.percentile(values, p, method="linear")) if len(values) >= minimum else None
        )
    result["warning"] = "small n; p99 is unstable" if len(values) < 100 else None
    return result


def lag_summary(values: list[float], threshold_ms: float, minimum: int = 2) -> dict[str, object]:
    result = percentiles(values, minimum)
    result["warning_threshold_ms"] = threshold_ms
    result["rate_not_maintained"] = result["p95"] is not None and result["p95"] > threshold_ms
    return result


def cost(rows: list[ResultRow], prices: dict[str, object]) -> dict[str, object]:
    n = sum(r.status == "ok" for r in rows)
    total = 0.0
    dates = set()
    for row in rows:
        if not row.usage_complete:
            return {"value": None, "reason": "incomplete usage", "n": n}
        price = prices.get("prices", {}).get(row.model, {})
        required = ("date", "input_per_million", "cached_input_per_million", "output_per_million")
        missing = [key for key in required if price.get(key) is None]
        if missing:
            return {"value": None, "reason": f"{row.model}: missing {', '.join(missing)}", "n": n}
        try:
            date.fromisoformat(str(price["date"]))
            if any(not np.isfinite(float(price[k])) or float(price[k]) < 0 for k in required[1:]):
                raise ValueError("Invalid price")
        except (ValueError, TypeError):
            return {"value": None, "reason": "invalid price/date", "n": n}
        dates.add(str(price["date"]))
        total += (
            (row.tokens_in - row.tokens_cached) * float(price["input_per_million"])
            + row.tokens_cached * float(price["cached_input_per_million"])
            + row.tokens_out * float(price["output_per_million"])
        ) / 1_000_000
    return {
        "value": total * 1000 / n if n else None,
        "n": n,
        "price_dates": sorted(dates),
        "billed": 0.0
        if rows and all(r.source == "real" and r.free_allocation is True for r in rows)
        else None,
    }


def scenario_metrics(
    rows: list[ResultRow], prices: dict[str, object], *, minimum: int = 2, lag_threshold: float = 20
) -> dict[str, object]:
    sent = len(rows)
    ok = [r for r in rows if r.status == "ok"]
    starts = [r.scheduled_arrival_monotonic for r in rows]
    ends = [r.completion_monotonic for r in rows]
    span = (max(starts) - min(starts)) / 1e9 if sent and None not in starts else None
    duration = (
        (max(ends) - min(starts)) / 1e9
        if sent and None not in starts and None not in ends
        else None
    )
    backed = [r for r in ok if r.provider_calls and not r.cache_hit]
    tokens_known = all(r.usage_complete for r in backed)
    queue = [r.queue_len_at_arrival for r in rows if r.queue_len_at_arrival is not None]
    high = [r.queue_high_water for r in rows if r.queue_high_water is not None]
    duplicate = [
        r.peak_duplicate_provider_operations
        for r in rows
        if r.peak_duplicate_provider_operations is not None
    ]
    return {
        "sent": sent,
        "duration_s": duration,
        "offered_per_minute": sent * 60 / span if span else None,
        "one_shot": bool(sent and all(r.arrival_kind == "one_shot" for r in rows)),
        "first_token_ms": percentiles(
            [r.first_token_ms for r in ok if r.first_token_ms is not None], minimum
        ),
        "total_ms": percentiles([r.total_ms for r in ok], minimum),
        "failed_elapsed_ms": [r.total_ms for r in rows if r.status != "ok"],
        "outcomes": {
            s: sum(r.status == s for r in rows) * 100 / sent if sent else None
            for s in ("ok", "rate_limited", "timeout", "error")
        },
        "retries_per_request": sum(r.retries for r in rows) / sent
        if sent and all(r.retries is not None for r in rows)
        else None,
        "peak_queue": max(queue) if queue else None,
        "queue_high_water": max(high) if high else None,
        "peak_duplicate_provider_operations": max(duplicate) if duplicate else None,
        "tokens_per_answer": sum(r.tokens_in + r.tokens_out for r in backed) / len(backed)
        if backed and tokens_known
        else None,
        "token_denominator": len(backed),
        "usage_complete": all(r.usage_complete for r in rows),
        "provider_calls_total": sum(r.provider_calls or 0 for r in rows),
        "retries_total": sum(r.retries or 0 for r in rows),
        "known_tokens_in_total": sum(r.known_tokens_in or 0 for r in rows),
        "known_tokens_out_total": sum(r.known_tokens_out or 0 for r in rows),
        "retry_wait_ms_total": sum(r.retry_wait_ms or 0 for r in rows),
        "mean_counted_tokens_per_answer": sum(r.counted_tokens for r in backed) / len(backed)
        if backed and all(r.counted_tokens is not None for r in backed)
        else None,
        "cost_per_1000": cost(rows, prices),
        "lag_ms": lag_summary(
            [r.generator_lag_ms for r in rows if r.generator_lag_ms is not None],
            lag_threshold,
            minimum,
        ),
    }


def capacity_point(
    rows: list[ResultRow],
    rate: float,
    deadline_ms: int,
    minimum_success: float,
    minimum_samples: int,
) -> dict[str, object]:
    metrics = scenario_metrics(rows, {}, minimum=minimum_samples)
    reasons = []
    if len(rows) < minimum_samples:
        reasons.append("insufficient samples")
    if not rows or metrics["outcomes"]["ok"] < minimum_success * 100:
        reasons.append("success fraction below threshold")
    p95 = metrics["total_ms"]["p95"]
    if p95 is None or p95 > deadline_ms:
        reasons.append("p95 unavailable or exceeds deadline")
    return {
        "rate_per_minute": rate,
        "passes": not reasons,
        "reasons": reasons,
        "generator_warning": metrics["lag_ms"]["rate_not_maintained"],
        "metrics": metrics,
    }


def capacity_result(points: list[dict[str, object]]) -> dict[str, object]:
    passing = [p["rate_per_minute"] for p in points if p["passes"]]
    highest = max(passing) if passing else None
    higher_tested = highest is not None and any(p["rate_per_minute"] > highest for p in points)
    return {
        "value": highest,
        "censored": highest is not None and not higher_tested,
        "interpretation": "lower bound; no higher point tested"
        if highest is not None and not higher_tested
        else "highest passing tested rate"
        if highest is not None
        else UNKNOWN,
    }


def limit_evidence(
    probes: list[dict[str, object]], documented: dict[str, object]
) -> list[list[str]]:
    result = []
    mapping = {"rpd": "x-ratelimit-limit-requests", "tpm": "x-ratelimit-limit-tokens"}
    for name in ("rpm", "tpm", "rpd", "tpd"):
        observations = []
        for record in probes:
            value = record.get("headers", {}).get(mapping.get(name, ""))
            if record.get("source") == "real" and record.get("dispatched") and value is not None:
                observations.append(
                    f"{value} ({record.get('observed_at_utc', UNKNOWN)}; "
                    f"{record.get('response_model', UNKNOWN)})"
                )
        saved = documented.get(name)
        if (
            isinstance(saved, dict)
            and saved.get("provenance") == "verified Limits page"
            and saved.get("value")
        ):
            observations.append(
                f"{saved['value']} (documented; {saved.get('date', UNKNOWN)}; "
                f"{saved.get('model', UNKNOWN)})"
            )
        result.append(
            [
                name.upper(),
                "; ".join(sorted(set(observations))) or UNKNOWN,
                "observed/documented" if observations else UNKNOWN,
            ]
        )
    return result


def show(value: object) -> str:
    return UNKNOWN if value is None else f"{value:.1f}" if isinstance(value, float) else str(value)


def evaluation_metrics(rows: list[ResultRow], minimum: int = 2) -> dict[str, dict[str, object]]:
    """Only explicit human grades enter accuracy denominators; never create grades."""
    result = {}
    for setting in ("A", "B", "C"):
        graded = [
            r
            for r in rows
            if r.record_kind == "eval" and r.retrieval_setting == setting and r.answer_grade
        ]
        n = len(graded)
        unanswerable = [r for r in graded if r.answerable is False]
        known_classes = all(r.answerable is not None for r in graded)
        result[setting] = {
            "asked": n,
            "correct_pct": sum(r.answer_grade == "correct" for r in graded) * 100 / n
            if n
            else None,
            "wrong_pct": sum(r.answer_grade == "wrong" for r in graded) * 100 / n if n else None,
            "unanswerable_n": len(unanswerable) if known_classes else None,
            "refused_correctly_pct": sum(
                r.answer_grade == "refused_correctly" for r in unanswerable
            )
            * 100
            / len(unanswerable)
            if known_classes and unanswerable
            else None,
            "performance": scenario_metrics(graded, {}, minimum=minimum),
        }
    return result


def distribution(value: dict[str, object]) -> str:
    return "/".join(show(value[f"p{p}"]) for p in (50, 95, 99)) + f" (n={value['n']})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def render(
    rows: list[ResultRow],
    prices: dict[str, object],
    probes: list[dict[str, object]],
    documented: dict[str, object],
    *,
    minimum: int,
    lag_threshold: float,
) -> tuple[str, dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        if row.record_kind == "load":
            groups[(row.fixture_only, row.scenario, row.source, row.run_id)].append(row)
    diagnostics = {}
    latency, outcomes, development = [], [], []
    measured = set()
    for (fixture, scenario, source, run_id), group in sorted(groups.items()):
        key = f"{scenario}/{source}/{run_id}"
        m = scenario_metrics(group, prices, minimum=minimum, lag_threshold=lag_threshold)
        diagnostics[key] = m
        if fixture:
            development.append(
                [
                    key,
                    str(m["sent"]),
                    distribution(m["total_ms"]),
                    json.dumps(m["outcomes"], sort_keys=True),
                ]
            )
            continue
        measured.add(scenario)
        offered = f"burst of {m['sent']}" if m["one_shot"] else show(m["offered_per_minute"])
        latency.append(
            [
                key,
                str(m["sent"]),
                show(m["duration_s"]),
                offered,
                distribution(m["first_token_ms"]),
                distribution(m["total_ms"]),
            ]
        )
        o = m["outcomes"]
        c = m["cost_per_1000"]
        displayed_cost = UNKNOWN if c["value"] is None else f"{c['value']:.4f} list-equiv"
        if c["value"] is not None and c.get("billed") == 0:
            displayed_cost = f"0.00 billed / {c['value']:.4f} list-equiv"
        # A free-tier billing claim requires explicit allocation evidence outside this CSV.
        outcomes.append(
            [
                key,
                show(o["ok"]),
                show(o["rate_limited"]),
                show(o["timeout"]),
                show(o["error"]),
                show(m["retries_per_request"]),
                show(m["peak_queue"]),
                f"{show(m['tokens_per_answer'])} (n={m['token_denominator']})",
                displayed_cost,
            ]
        )
    for scenario in OFFICIAL:
        if scenario not in measured:
            latency.append([scenario] + [UNKNOWN] * 5)
            outcomes.append([scenario] + [UNKNOWN] * 7)
    text = "# Relay experimental results\n\nTable 1 — real account evidence\n\n"
    text += table(["Limit", "Evidence", "Basis"], limit_evidence(probes, documented))
    text += "\n\nTable 2a — successful client latency (ms; linear p50/p95/p99)\n\n"
    text += table(
        ["Scenario/source/run", "Sent", "Duration s", "Offered/min", "First token", "Total"],
        latency,
    )
    text += "\n\nTable 2b — outcomes\n\n"
    text += table(
        [
            "Scenario/source/run",
            "OK %",
            "Limited %",
            "Timeout %",
            "Retries/request",
            "Peak queue",
            "Tokens/answer",
            "Cost/1000",
        ],
        outcomes,
    )
    text += (
        "\n\nError percentages, failed elapsed times, usage completeness, price availability, "
        "queue high-water and generator lag are recorded per run in table_diagnostics.json. "
        "Tokens include cached input; the denominator excludes cache hits "
        "and includes validated refusals. Cost includes all attempts, including failures; "
        "billing is not inferred without allocation evidence. "
        "Percentiles need the configured minimum sample count; p99 with n<100 is unstable.\n"
    )
    evaluation = evaluation_metrics([r for r in rows if not r.fixture_only], minimum)
    diagnostics["evaluation"] = evaluation
    text += "\nTable 3 — manual retrieval evaluation\n\n"
    evaluation_rows = []
    for setting, e in evaluation.items():
        m = e["performance"]
        evaluation_rows.append(
            [
                setting,
                str(e["asked"]),
                show(e["correct_pct"]),
                show(e["wrong_pct"]),
                f"{show(e['refused_correctly_pct'])} (n={show(e['unanswerable_n'])})",
                f"{show(m['tokens_per_answer'])} (n={m['token_denominator']})",
                distribution(m["first_token_ms"]),
                distribution(m["total_ms"]),
            ]
        )
    text += table(
        [
            "Setting",
            "Graded questions",
            "Correct/cited %",
            "Wrong %",
            "Correct refusal %",
            "Tokens/answer",
            "First ms p50/p95/p99",
            "Total ms p50/p95/p99",
        ],
        evaluation_rows,
    )
    text += "\nFixture-only development diagnostics (excluded from official tables)\n\n"
    text += table(["Scenario/source/run", "Sent", "Total ms p50/p95/p99", "Outcome %"], development)
    return text + "\n", diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--prices", type=Path, default=Path("config/prices.yaml"))
    parser.add_argument("--probes", type=Path, default=Path("results/limit_probe.jsonl"))
    parser.add_argument("--limits", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tables.md"))
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    if args.submission:
        root = Path("results/p7")
        rows = read_csv(root / "results.csv")
        account = verified_account_evidence(
            Path("results/final_completion/account_evidence/manifest.json")
        )
        data, diagnostics = submission_tables(rows, account=account)
        root.joinpath("tables.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        root.joinpath("metrics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        root.joinpath("tables.md").write_text(
            "\n\n".join(name + "\n\n" + table(t["headers"], t["rows"]) for name, t in data.items()),
            encoding="utf-8",
        )
        print("Generated submission tables from separate validated P7 CSV")
        return 0
    settings = load_settings()
    rows = read_csv(args.input)  # Explicit input only; never search for stale CSV files.
    prices = yaml.safe_load(args.prices.read_text(encoding="utf-8"))
    probes = json_rows([args.probes]) if args.probes.exists() else []
    limits = json.loads(args.limits.read_text()) if args.limits else {}
    content, diagnostics = render(
        rows,
        prices,
        probes,
        limits,
        minimum=settings.min_percentile_samples,
        lag_threshold=settings.lag_warning_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    args.output.with_name("table_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"Calculated tables from {len(rows)} validated logical rows")
    return 0


def human_agreement(grades: list[dict[str, object]]) -> dict[str, object]:
    grouped = defaultdict(list)
    for grade in grades:
        grouped[grade["request_id"]].append(grade)
    paired = [
        sorted(group, key=lambda g: int(g["round"])) for group in grouped.values() if len(group) > 1
    ]
    return {
        "n": len(paired),
        "agreement": sum(g[0]["answer_grade"] == g[-1]["answer_grade"] for g in paired)
        / len(paired)
        if paired
        else None,
    }


def quality(group: list[ResultRow]) -> dict[str, object]:
    graded = [r for r in group if r.answer_grade]
    answerable = [r for r in graded if r.answerable]
    absent = [r for r in graded if r.answerable is False]
    return {
        "asked": len(group),
        "graded": len(graded),
        "ungraded": len(group) - len(graded),
        "uncertain": sum(r.review_uncertain is True for r in group),
        "assistant_reviewed": sum(r.grading_provenance == "assistant_review" for r in graded),
        "grade_counts": {
            name: sum(r.answer_grade == name for r in graded)
            for name in ["correct", "wrong", "refused_correctly", "refused_wrongly"]
        },
        "answerable": sum(r.answerable is True for r in group),
        "unanswerable": sum(r.answerable is False for r in group),
        "correct_pct": sum(r.answer_grade == "correct" for r in graded) * 100 / len(graded)
        if graded
        else None,
        "wrong_pct": sum(r.answer_grade == "wrong" for r in graded) * 100 / len(graded)
        if graded
        else None,
        "answerable_accuracy": sum(r.answer_grade == "correct" for r in answerable)
        * 100
        / len(answerable)
        if answerable
        else None,
        "answerable_graded_n": len(answerable),
        "unanswerable_graded_n": len(absent),
        "refusal_accuracy": sum(r.answer_grade == "refused_correctly" for r in absent)
        * 100
        / len(absent)
        if absent
        else None,
        "overall_accuracy": sum(r.answer_grade in {"correct", "refused_correctly"} for r in graded)
        * 100
        / len(graded)
        if graded
        else None,
        "citation_valid_n": sum(r.citation_valid is True for r in group),
        "citation_observed_n": sum(r.citation_valid is not None for r in group),
        "retrieval_hit_n": sum(r.retrieval_hit is True for r in group),
        "retrieval_observed_n": sum(r.retrieval_hit is not None for r in group),
        "performance": scenario_metrics(group, {}),
    }


def verified_account_evidence(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    evidence = json.loads(path.read_text())
    for entry in evidence["screenshots"]:
        artifact = (path.parent / entry["filename"]).resolve()
        if (
            not artifact.is_relative_to(path.parent.resolve())
            or not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Account screenshot differs from reviewed evidence")
    return evidence


def submission_tables(
    rows: list[ResultRow],
    connectivity: dict[str, object] | None = None,
    account: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    connectivity = (
        connectivity
        or json.loads(Path("results/connectivity/first_real_result.json").read_text())["observed"]
    )
    evidence = limit_evidence([connectivity], {})
    names = ["Requests per minute", "Tokens per minute", "Requests per day", "Tokens per day"]
    t1 = [
        [
            name,
            e[1],
            "Timestamped real response header" if e[1] != UNKNOWN else UNKNOWN,
            "high for observed header" if e[1] != UNKNOWN else UNKNOWN,
        ]
        for name, e in zip(names, evidence, strict=True)
    ]
    console = yaml.safe_load(Path("config/p7.yaml").read_text())["limits"]
    for index, value in [(0, console["rpm"]), (3, console["tpd"])]:
        t1[index][1:] = [
            str(value),
            "Candidate-reported organization console limit.",
            "unverified; screenshot missing",
        ]
    if account:
        for index, field in [(0, "rpm"), (3, "tpd")]:
            t1[index][1:] = [
                str(account["organization_limits"][field]),
                "Organization Limits screenshot; file dated 2026-11-14; capture time unverified",
                "high for displayed limit",
            ]
    request_id = connectivity.get("provider_request_id")
    if request_id:
        for index in [1, 2]:
            t1[index][2] += f"; request {request_id}"
    t1 += [
        [name, UNKNOWN, UNKNOWN, UNKNOWN]
        for name in [
            "Requests in flight at once",
            "What it returns when a limit is hit",
            "Time until it recovers",
        ]
    ]
    latency = []
    outcomes = []
    diagnostics = {}
    for name in ["eval_real", "real_burst6", *OFFICIAL, "real_smoke"]:
        group = [r for r in rows if r.scenario == name and not r.fixture_only]
        development = [r for r in rows if r.scenario == "development_" + name]
        label = name
        if not group and development:
            selected_run = max(development, key=lambda r: r.start_time).run_id
            group = [r for r in development if r.run_id == selected_run]
            diagnostics[name + "/selected_run"] = selected_run
            for old_run in sorted({r.run_id for r in development}):
                diagnostics[name + "/run/" + old_run] = scenario_metrics(
                    [r for r in development if r.run_id == old_run], {}
                )
            label = name + " (dev)"
        source = "real" if name in {"eval_real", "real_burst6", "real_smoke"} else "stand-in"
        if not group:
            latency.append([name, source] + [UNKNOWN] * 5)
            outcomes.append([name] + [UNKNOWN] * 8)
            continue
        m = scenario_metrics(group, {})
        diagnostics[name] = m
        latency.append(
            [
                label,
                source,
                str(m["sent"]),
                show(
                    m["duration_s"]
                    if m["duration_s"] is not None
                    else group[0].total_ms / 1000
                    if len(group) == 1
                    else None
                ),
                show(m["offered_per_minute"]),
                distribution(m["first_token_ms"]),
                distribution(m["total_ms"]),
            ]
        )
        o = m["outcomes"]
        outcomes.append(
            [
                label,
                show(o["ok"]),
                show(o["rate_limited"]),
                show(o["timeout"]),
                show(o["error"]),
                show(m["retries_per_request"]),
                show(m["peak_queue"]),
                f"{show(m['tokens_per_answer'])} (n={m['token_denominator']})",
                UNKNOWN,
            ]
        )
    table3 = []
    evaluation = [r for r in rows if r.record_kind == "eval" and r.question_id]
    settings = {
        "A": "256, 6, no",
        "B": "512, 4, no",
        "C": "384, 12 to 4, actual rerank in run diagnostics",
    }
    for source in ["real", "stand-in"]:
        source_rows = [r for r in evaluation if r.source == source]
        latest = max(source_rows, key=lambda r: r.start_time).run_id if source_rows else None
        diagnostics[f"evaluation/{source}/selected_run"] = latest
        for run_id in sorted({r.run_id for r in source_rows}):
            diagnostics[f"evaluation/{source}/run/{run_id}"] = quality(
                [r for r in source_rows if r.run_id == run_id]
            )
        for setting in ["A", "B", "C"]:
            group = [r for r in evaluation if r.source == source and r.retrieval_setting == setting]
            group = [r for r in group if r.run_id == latest]
            q = quality(group)
            m = q["performance"]
            diagnostics[f"evaluation/{source}/{setting}"] = q
            table3.append(
                [
                    f"{setting}: {settings[setting]} ({source})",
                    str(q["asked"]),
                    f"{show(q['correct_pct'])} (graded n={q['graded']})",
                    f"{show(q['wrong_pct'])} (graded n={q['graded']})",
                    f"{show(q['refusal_accuracy'])} (graded n={q['unanswerable_graded_n']})",
                    f"{show(m['tokens_per_answer'])} (n={m['token_denominator']})",
                    "/".join(show(m["first_token_ms"][f"p{p}"]) for p in [50, 95])
                    + f" (n={m['first_token_ms']['n']})",
                    "/".join(show(m["total_ms"][f"p{p}"]) for p in [50, 95])
                    + f" (n={m['total_ms']['n']})",
                ]
            )
            for paper in ["transformer", "moshi", "dynamic_chunking"]:
                diagnostics[f"evaluation/{source}/{setting}/{paper}"] = quality(
                    [r for r in group if r.paper_id == paper]
                )
    review_groups = [diagnostics[f"evaluation/stand-in/{setting}"] for setting in ["A", "B", "C"]]
    diagnostics["review_summary"] = {
        "asked": sum(q["asked"] for q in review_groups),
        "reviewed": sum(q["graded"] for q in review_groups),
        "uncertain": sum(q["uncertain"] for q in review_groups),
        "provenance": "assistant_review; candidate confirmation pending",
    }
    return {
        "Table 1: Limits found in the real service": {
            "headers": ["Limit", "Value you found", "How you found it", "Confidence"],
            "rows": t1,
        },
        "Table 2a: Latency": {
            "headers": [
                "Scenario",
                "Real or stand-in",
                "Requests sent",
                "Duration (s)",
                "Offered rate (/min)",
                "First token p50 / p95 / p99 (ms)",
                "Total p50 / p95 / p99 (ms)",
            ],
            "rows": latency,
        },
        "Table 2b: Outcomes": {
            "headers": [
                "Scenario",
                "Succeeded %",
                "Rate-limited %",
                "Timed out or given up %",
                "Error %",
                "Retries per request",
                "Peak queue length",
                "Tokens per answer",
                "Cost per 1,000 answers",
            ],
            "rows": outcomes,
        },
        "Table 3: Retrieval settings": {
            "headers": [
                "Setting (piece size, pieces fetched, re-rank)",
                "Questions asked",
                "Correct and cited %",
                "Wrong %",
                "Correctly refused %",
                "Tokens per answer",
                "First token p50 / p95 (ms)",
                "Total p50 / p95 (ms)",
            ],
            "rows": table3,
        },
    }, diagnostics


if __name__ == "__main__":
    raise SystemExit(main())
