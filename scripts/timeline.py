"""Partition measured request intervals into exclusive waiting/working/retrying time."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loadgen.models import ClientRecord
from scripts.build_results_csv import json_rows


def decompose(client: ClientRecord, events: list[dict[str, object]]) -> dict[str, object]:
    total = client.total_ms
    missing = []
    arrival = next((e for e in events if e.get("ev") == "arrived"), {})
    absolute = arrival.get("server_arrival_monotonic")
    aligned = isinstance(absolute, int)
    shift = (absolute - client.actual_send_monotonic) / 1e6 if aligned else 0
    if not aligned:
        missing.append(
            "server arrival absolute monotonic alignment; server-local spans shown separately"
        )
    intervals: list[tuple[float, float, str, str]] = []

    def add(start: object, end: object, category: str, label: str) -> None:
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            missing.append(f"{label}: start/end timestamps")
            return
        if end < start:
            raise ValueError(f"Invalid event interval: {label}")
        a, b = max(0, start + shift), min(total, end + shift)
        if b > a:
            intervals.append((a, b, category, label))

    calls = defaultdict(list)
    for event in events:
        kind = event.get("ev")
        if kind == "execution_start":
            end, duration = event.get("t_ms"), event.get("queue_wait_ms")
            add(
                end - duration
                if isinstance(end, (int, float)) and isinstance(duration, (int, float))
                else None,
                end,
                "waiting",
                "logical queue",
            )
        elif kind in {"provider_wait", "retry_wait"}:
            add(
                event.get("start_ms"),
                event.get("end_ms"),
                "retrying" if kind == "retry_wait" else "waiting",
                str(kind),
            )
        elif kind == "stage_timing":
            # Draft/verifier stages contain scheduler and retry waits; call spans below
            # distinguish actual provider work without assuming unrecorded waits are zero.
            if event.get("stage") not in {"draft", "verification", "repair"}:
                add(event.get("start_ms"), event.get("end_ms"), "working", str(event.get("stage")))
        if event.get("call_id") and kind in {"call_start", "call_end", "call_error"}:
            calls[event["call_id"]].append(event)
    for cid, records in calls.items():
        start = next((e for e in records if e["ev"] == "call_start"), None)
        end = next((e for e in records if e["ev"] in {"call_end", "call_error"}), None)
        if not start or not end:
            missing.append(f"call {cid}: start/terminal event")
            continue
        category = "retrying" if start.get("attempt", 1) > 1 else "working"
        add(
            start.get("t_ms"),
            end.get("t_ms"),
            category,
            f"{start.get('kind')} attempt {start.get('attempt', 1)}",
        )
    if not any(e.get("ev") == "execution_start" for e in events):
        missing.append("execution_start (may be an admission rejection)")
    if not any(e.get("ev") == "stage_timing" for e in events):
        missing.append("stage_timing")
    if any(e.get("ev") == "retry_attempt" for e in events) and not any(
        e.get("ev") == "retry_wait" for e in events
    ):
        missing.append("retry_wait actual interval")
    points = sorted({0.0, total, *(t for a, b, _, _ in intervals for t in (a, b))})
    totals = dict.fromkeys(("waiting", "working", "retrying", "unknown"), 0.0)
    partition = []
    priority = {"working": 0, "waiting": 1, "retrying": 2}
    for a, b in zip(points, points[1:], strict=False):
        covered = [category for start, end, category, _ in intervals if start <= a and end >= b]
        category = max(covered, key=priority.get) if covered and aligned else "unknown"
        totals[category] += b - a
        partition.append({"start_ms": a, "end_ms": b, "category": category})
    first = client.first_token_ms
    after = total - first if first is not None else None
    content = next(
        (
            e.get("provider_elapsed_ms")
            for e in events
            if e.get("ev") == "first_delta" and e.get("what") == "content"
        ),
        None,
    )
    return {
        "request_id": client.request_id,
        "client_status": client.status,
        "total_client_ms": total,
        **{k + "_ms": v for k, v in totals.items()},
        "provider_ttft_ms": content,
        "user_visible_first_token_ms": first,
        "before_first_token_ms": first,
        "after_first_token_ms": after,
        "principal_delay": "unknown" if first is None else "before" if first >= after else "after",
        "alignment_known": aligned,
        "missing_events": sorted(set(missing)),
        "exclusive_partition": partition,
        "recorded_spans": [
            {"start_ms": a, "end_ms": b, "category": c, "label": label}
            for a, b, c, label in intervals
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_id")
    parser.add_argument("--events", type=Path, default=Path("results/events.jsonl"))
    parser.add_argument("--clients", type=Path, default=Path("results/client_rows.jsonl"))
    args = parser.parse_args()
    clients = [r for r in json_rows([args.clients]) if r.get("request_id") == args.request_id]
    if len(clients) != 1:
        parser.error("Expected exactly one logical client result for request ID")
    events = [
        e
        for e in json_rows([args.events])
        if (e.get("rid") or e.get("request_id")) == args.request_id
    ]
    print(json.dumps(decompose(ClientRecord.model_validate(clients[0]), events), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
