"""Join logical client/eval rows with call-ID events; never calculate table figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from loadgen.models import ClientRecord
from relay.obs.events import Sanitizer
from relay.providers.base import Usage
from scripts.result_models import COLUMNS, EXTRA, REQUIRED, ResultRow


class ResultsError(ValueError):
    pass


def json_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Object required")
                    rows.append(value)
                except ValueError:
                    raise ResultsError(f"{path.name}:{number}: malformed JSON object") from None
    return rows


def events_by_request(events: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped = defaultdict(list)
    owners = {}
    for event in events:
        rid = event.get("rid") or event.get("request_id")
        if not isinstance(rid, str) or not rid:
            raise ResultsError("<event>: missing request identifier")
        cid = event.get("call_id")
        if cid:
            if cid in owners and owners[cid] != rid:
                raise ResultsError(f"{rid}: call ID has multiple logical owners")
            owners[cid] = rid
        grouped[rid].append(event)
    return dict(grouped)


def call_metrics(rid: str, events: list[dict[str, object]]) -> dict[str, object]:
    calls = defaultdict(list)
    for event in events:
        if event.get("ev") in {"call_start", "call_end", "call_error"} and event.get("call_id"):
            calls[event["call_id"]].append(event)
    usages = []
    elapsed = []
    retries = 0
    diagnostics = []
    operations: dict[str, list[tuple[float, int]]] = defaultdict(list)
    overlap_known = True
    for cid, records in calls.items():
        starts = [e for e in records if e["ev"] == "call_start"]
        ends = [e for e in records if e["ev"] in {"call_end", "call_error"}]
        if len(starts) > 1 or len(ends) > 1:
            raise ResultsError(f"{rid}: duplicate call start/terminal {cid}")
        end = ends[0] if ends else None
        if (
            end
            and isinstance(end.get("safe_context"), dict)
            and end["safe_context"].get("dispatched") is False
        ):
            continue
        if (
            starts
            and end
            and isinstance(starts[0].get("t_ms"), (int, float))
            and isinstance(end.get("t_ms"), (int, float))
        ):
            if end["t_ms"] < starts[0]["t_ms"]:
                raise ResultsError(f"{rid}: provider completion precedes call start")
            operations[str(starts[0].get("kind"))].extend(
                [(starts[0]["t_ms"], 1), (end["t_ms"], -1)]
            )
        else:
            overlap_known = False
        if not starts and not end:
            continue
        if not starts or not end:
            diagnostics.append("incomplete_call_events")
        try:
            usages.append(Usage.model_validate(end.get("usage", {}) if end else {}))
        except ValueError:
            raise ResultsError(f"{rid}: malformed provider usage for {cid}") from None
        elapsed.append(end.get("total_ms") if end else None)
        if elapsed[-1] is not None and (
            not isinstance(elapsed[-1], (float, int)) or elapsed[-1] < 0
        ):
            raise ResultsError(f"{rid}: invalid provider duration for {cid}")
        retries += bool(starts and starts[0].get("attempt", 1) > 1)
    if not usages:
        return {}
    result: dict[str, object] = {
        "provider_calls": len(usages),
        "usage_complete": all(u.complete for u in usages),
        "retries": retries,
        "accounting_diagnostics": diagnostics,
    }
    peak_duplicate = 0
    for boundaries in operations.values():
        active = 0
        for _, increment in sorted(boundaries):
            active += increment
            peak_duplicate = max(peak_duplicate, active - 1)
    result["peak_duplicate_provider_operations"] = peak_duplicate if overlap_known else None
    for field in ("tokens_in", "tokens_out", "tokens_cached", "counted_tokens"):
        values = [getattr(u, field) for u in usages]
        result[field] = sum(values) if all(v is not None for v in values) else None
        result["known_" + field] = sum(v for v in values if v is not None)
    result["provider_ms"] = (
        sum(elapsed) if all(isinstance(v, (float, int)) for v in elapsed) else None
    )
    content = [e for e in events if e.get("ev") == "first_delta" and e.get("what") == "content"]
    result["provider_ttft_ms"] = content[0].get("provider_elapsed_ms") if content else None
    attempts = [e for e in events if e.get("ev") == "retry_attempt"]
    if attempts:
        result["retries"] = len(attempts)
    return result


def joined(
    client: ClientRecord, events: list[dict[str, object]], manifest: dict[str, object] | None = None
) -> ResultRow:
    data = client.model_dump() | {
        "start_time": client.actual_send_utc,
        "client_status": client.status,
        "answer_grade": "",
    }
    if manifest:
        for key in ("run_id", "scenario", "source", "model"):
            if manifest.get(key) is not None and manifest[key] != data[key]:
                raise ResultsError(f"{client.request_id}: manifest {key} mismatch")
        data["server_cleanup_observed"] = manifest.get("server_cleanup_observed")
        data["arrival_kind"] = manifest.get("offered_arrivals", {}).get("kind")
    metrics = call_metrics(client.request_id, events)
    if metrics:
        if client.provider_calls is not None and client.provider_calls != metrics["provider_calls"]:
            raise ResultsError(f"{client.request_id}: provider-call count mismatch")
        data.update(metrics)
    elif (
        client.provider_calls == 0
        or any(e.get("ev") == "done" and e.get("provider_calls") == 0 for e in events)
        or any(e.get("ev") == "rejected" for e in events)
    ):
        data.update(
            {
                "provider_calls": 0,
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_cached": 0,
                "counted_tokens": 0,
                "usage_complete": True,
            }
        )
    else:
        data["accounting_diagnostics"] = ["provider_accounting_unavailable"]
    stages = [e for e in events if e.get("ev") == "stage_timing"]
    for field, names in (
        ("retrieval_ms", {"retrieval"}),
        ("validation_ms", {"verification", "deterministic_validation"}),
    ):
        selected = [e["duration_ms"] for e in stages if e.get("stage") in names]
        data[field] = sum(selected) if selected else None
    waits = [e.get("duration_ms") for e in events if e.get("ev") == "retry_wait"]
    data["retry_wait_ms"] = sum(waits) if waits else 0 if data.get("retries") == 0 else None
    arrival = next((e for e in events if e.get("ev") == "arrived"), {})
    admitted = next((e for e in events if e.get("ev") == "admitted"), {})
    execution = next((e for e in events if e.get("ev") == "execution_start"), {})
    data["queue_len_at_arrival"] = (admitted or arrival).get("queue_length")
    data["inflight_at_arrival"] = arrival.get("inflight")
    data["queue_wait_ms"] = execution.get(
        "queue_wait_ms", 0 if any(e.get("ev") == "rejected" for e in events) else None
    )
    data["mode"] = admitted.get("mode", data.get("mode") or arrival.get("mode"))
    peaks = [e["peak_queue_length"] for e in events if isinstance(e.get("peak_queue_length"), int)]
    data["queue_high_water"] = max(peaks) if peaks else None
    validated = [e for e in events if e.get("ev") == "validated"]
    if validated and client.status == "ok":
        data["citation_valid"] = (
            validated[-1].get("det") == "pass" and validated[-1].get("llm") == "pass"
        )
    return ResultRow.model_validate(data)


def build(
    client_paths: list[Path],
    event_paths: list[Path],
    manifest_paths: list[Path],
    eval_paths: list[Path] | None = None,
) -> list[ResultRow]:
    grouped = events_by_request(json_rows(event_paths))
    manifests = {}
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("run_id") in manifests:
            raise ResultsError("Duplicate run manifest")
        manifests[manifest.get("run_id")] = manifest
    result = []
    seen = set()
    errors = []
    for raw in json_rows(client_paths):
        rid = raw.get("request_id", "<missing>")
        try:
            if rid in seen:
                raise ValueError("duplicate logical request")
            seen.add(rid)
            client = ClientRecord.model_validate(raw)
            result.append(joined(client, grouped.get(rid, []), manifests.get(client.run_id)))
        except (ValueError, TypeError) as exc:
            reason = (
                str(exc)
                if not isinstance(exc, ValidationError)
                else str(exc.errors(include_input=False, include_url=False))
            )
            errors.append(f"{rid}: {reason}")
    for raw in json_rows(eval_paths or []):
        rid = raw.get("request_id", "<missing>")
        try:
            if rid in seen:
                raise ValueError("duplicate logical request across inputs")
            seen.add(rid)
            row = ResultRow.model_validate(raw | {"record_kind": "eval"})
            if row.fixture_only and row.answer_grade:
                raise ValueError("Fixture grades cannot enter results.csv")
            result.append(row)
        except ValueError as exc:
            detail = (
                exc.errors(include_input=False, include_url=False)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            errors.append(f"{rid}: {detail}")
    if errors:
        raise ResultsError("\n".join(errors))
    return sorted(result, key=lambda row: row.request_id)


def write_csv(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            data = row.model_dump()
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = json.dumps(value, separators=(",", ":"))
                elif isinstance(value, float):
                    data[key] = f"{value:.1f}"
            writer.writerow(data)
    read_csv(temporary)
    temporary.replace(path)


def read_csv(path: Path) -> list[ResultRow]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if (
            not reader.fieldnames
            or reader.fieldnames[: len(REQUIRED + EXTRA)] != REQUIRED + EXTRA
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
        ):
            raise ResultsError("<header>: missing or misordered mandatory columns")
        result = []
        seen = set()
        errors = []
        for raw in reader:
            rid = raw.get("request_id") or "<missing>"
            try:
                if rid in seen:
                    raise ValueError("duplicate request identifier")
                seen.add(rid)
                data = {key: value if value != "" else None for key, value in raw.items()}
                data["answer_grade"] = raw.get("answer_grade") or ""
                data["accounting_diagnostics"] = json.loads(
                    raw.get("accounting_diagnostics") or "[]"
                )
                row = ResultRow.model_validate(data)
                if row.fixture_only and row.answer_grade:
                    raise ValueError("Fixture grades cannot enter results.csv")
                result.append(row)
            except (ValueError, TypeError) as exc:
                detail = (
                    exc.errors(include_input=False, include_url=False)
                    if isinstance(exc, ValidationError)
                    else str(exc)
                )
                errors.append(f"{rid}: {detail}")
        if errors:
            raise ResultsError("\n".join(errors))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients", type=Path, nargs="+", default=[Path("results/client_rows.jsonl")]
    )
    parser.add_argument("--events", type=Path, nargs="*", default=[Path("results/events.jsonl")])
    parser.add_argument("--manifests", type=Path, nargs="*")
    parser.add_argument("--eval-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    try:
        if args.validate:
            rows = read_csv(args.validate)
        else:
            manifests = (
                args.manifests
                if args.manifests is not None
                else sorted(Path("results/runs").glob("*/run_manifest.json"))
            )
            rows = build(
                args.clients, [p for p in args.events if p.exists()], manifests, args.eval_runs
            )
            write_csv(rows, args.output)
        print(f"Validated {len(rows)} logical rows")
        return 0
    except (ValueError, OSError) as exc:
        for line in str(exc).splitlines():
            print(Sanitizer().text(line, body=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
