"""Build separate submission CSV inputs; never modify historical P6 artifacts."""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from eval.dataset import load_key
from eval.grade import Grade
from scripts.build_results_csv import json_rows, read_csv, write_csv
from scripts.result_models import ResultRow


def convert(raw: dict[str, object], key: dict[str, object]) -> ResultRow:
    usage = raw.get("usage", {})
    data = raw.copy()
    data.update(
        record_kind="eval",
        answer_grade="",
        fixture_only=False,
        answerable=key.get("answerable"),
        paper_id=raw.get("paper"),
        scenario="eval_standin" if raw["source"] == "stand-in" else "eval_real",
        model="openai/gpt-oss-20b",
        usage_complete=usage.get("complete"),
        answer_disposition="refused"
        if raw.get("validated_refusal")
        else "answered"
        if raw["status"] == "ok"
        else "failed",
        first_token_ms=raw.get("first_token_ms") if raw["status"] == "ok" else None,
    )
    for field in (
        "tokens_in",
        "tokens_out",
        "tokens_cached",
        "counted_tokens",
        "known_tokens_in",
        "known_tokens_out",
        "known_tokens_cached",
        "known_counted_tokens",
    ):
        data[field] = usage.get(field)
    data["retries"] = raw.get("retries", 0)
    data["citation_valid"] = (
        (raw.get("validation") or {}).get("status") == "pass" if raw.get("citations") else None
    )
    return ResultRow.model_validate(data)


def build(root: Path = Path("results/p7")) -> list[ResultRow]:
    rows = read_csv(Path("results/results.csv"))
    keys = load_key()
    for path in sorted((root / "standin_evaluation").glob("eval-*/rows.jsonl")):
        for raw in json_rows([path]):
            rows.append(convert(raw, keys[raw["question_id"]].model_dump()))
    smoke = json.loads((root / "real_smoke/outcome.json").read_text())
    calls = smoke.get("ledger_calls", [])
    events = json_rows([root / "real_smoke/events.jsonl"])
    start = next(e for e in events if e.get("ev") == "call_start")
    stamp = datetime.fromisoformat(calls[0]["ts_utc"].replace("Z", "+00:00")) - timedelta(
        milliseconds=start["t_ms"]
    )
    smoke.update(
        source="real",
        retrieval_setting=smoke["setting"],
        start_time=stamp.isoformat(timespec="milliseconds"),
    )
    row = convert(smoke, {})
    row.scenario = "real_smoke"
    row.record_kind = "eval"  # Diagnostic only; no question_id, so excluded from Table 3.
    rows.append(row)
    grades_path = root / "grades.csv"
    if grades_path.exists():
        with grades_path.open(encoding="utf-8", newline="") as source:
            grades = [Grade.model_validate(g) for g in csv.DictReader(source)]
        by_id = {r.request_id: r for r in rows if r.question_id}
        for grade in grades:
            if grade.request_id not in by_id:
                raise ValueError("Grade does not reference a candidate evaluation result")
            by_id[grade.request_id].answer_grade = grade.answer_grade
    write_csv(rows, root / "results.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"Built {len(build())} separately retained P7 submission rows")
