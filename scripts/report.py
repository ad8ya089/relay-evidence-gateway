"""Render the exact protected write-up headings using generated tables and recorded evidence."""

import html
import json
import re
from pathlib import Path

import fitz

from relay.app.messages import message
from scripts.aggregate import table

ROOT = Path("results/p7")


def headings() -> list[str]:
    return re.findall(
        r"^### (.+)$", Path("docs/06_WRITEUP_TEMPLATE.md").read_text(encoding="utf-8"), re.M
    )


def report_parts() -> tuple[list[tuple[str, str]], dict[str, object]]:
    tables = json.loads((ROOT / "tables.json").read_text())
    metrics = json.loads((ROOT / "metrics.json").read_text())
    timeline = json.loads(Path("results/p6_timeline_example.json").read_text())
    texts = [
        "Relay owns interpretation, a retrieval tool, an evidence gate with bounded broadening, "
        "drafting, deterministic and model verification, repair and sealed release. The principal "
        "decision is to buffer answers until validation, accepting slower visible content rather "
        "than releasing unsupported text. Admission, deadlines, quotas, retries and "
        "caching remain application-owned.",
        "The guarded adapter now supports explicit Groq selection; stand-in remains the default. "
        "The connectivity request reached Groq but its requested marker was "
        "truncated. The end-to-end "
        "smoke reached Groq and safely abstained: its quote was not a substring of "
        "its cited chunk. "
        "Calibration has one fitted draft TTFT observation, no verifier samples and "
        "no complete paired "
        "latency distribution. It does not establish stand-in fidelity, concurrency "
        "or recovery behaviour.",
        "All percentiles use NumPy linear interpolation over status=ok only, with n "
        "shown; failed elapsed "
        "times remain in diagnostics. No real A/B/C quality run passed the "
        "conservative quota/session "
        "preflight. Local stand-in A/B/C workflow results are separate and "
        "ungraded. Development-profile local load runs are labelled (dev); "
        "official calibrated runs "
        "remain blocked. One unguessed observation: the real smoke was rejected for "
        "quote anchoring, "
        "not an HTTP failure; its full failure record is retained. Load rows have "
        "blank answer_grade. "
        f"The real smoke error share is {metrics['real_smoke']['outcomes']['error']:.1f}%; "
        "errors complete the outcome percentages. Latest corrected stand-in runs are shown; "
        "the earlier endpoint-error run remains in CSV and per-run diagnostics. "
        "Incomplete failed-attempt usage prevents complete token averages.",
        f"Measured fixture-only stand-in request {timeline['request_id']}: "
        f"waiting={timeline['waiting_ms']} ms; "
        f"working={timeline['working_ms']} ms; retrying={timeline['retrying_ms']} ms; "
        f"unattributed={timeline['unknown_ms']} ms. "
        f"First visible content={timeline['user_visible_first_token_ms']} ms; "
        f"after-first={timeline['after_first_token_ms']} ms. "
        "The principal delay was before release. "
        "This is a recorded P6 development trace, not Groq latency; raw event "
        "intervals and client rows are packaged.",
        "Predictive admission rejects work that cannot fit quota, queue or deadline "
        "bounds. Pressure mode "
        "reduces context and output limits; it never skips validation. Daily "
        "exhaustion permits eligible "
        "verified cache hits and stops new model work. In development, minute-limit "
        "halving did not bind at the tested rate; "
        "daily-limit reduction rejected later uncached work. "
        "Real recovery times remain not measured.",
        "A/B use lexical retrieval; C uses hybrid retrieval and attempts reranking, "
        "recording any fallback. "
        "No setting is selected on answer quality because human grading is pending. "
        "Performance is shown "
        "separately for stand-in execution. Reference anchors are literal overlaps "
        "with candidate answer "
        "notes, not a substitute for human citation support review. Gold references "
        "never enter model requests.",
        "Three observed A top-one traps: conversational wording in B01 retrieved "
        "a misspelling dialogue (moshi-v1-p64-c325, F.2), not system architecture; "
        "meaning and acoustic-information wording in B02 retrieved an entropy "
        "dialogue (moshi-v1-p62-c319, F.1), not Mimi's split quantizers; C03's "
        "training/compression wording retrieved an introductory overview "
        "(dynamic_chunking-v2-p2-c7), not the smoothing/ratio-loss sections. "
        "These are ranking mismatches, not proof that the full top-k set or "
        "final answer failed. A01's top hit was the correct attention section.",
        "One owner enforces bounded retries, jitter, deadlines and a global budget; "
        "HTTP retries are off. "
        "In paired uncalibrated development runs (same seed, workload, schedule and faults), "
        f"naive success was {metrics['retry_naive']['outcomes']['ok']:.1f}% versus "
        f"bounded {metrics['retry_bounded']['outcomes']['ok']:.1f}% "
        f"({metrics['retry_naive']['sent']} requests each). "
        f"Naive used {metrics['retry_naive']['provider_calls_total']} provider calls versus "
        f"{metrics['retry_bounded']['provider_calls_total']}; "
        "shielded slow work spawned duplicates. "
        "Naive successful latency percentiles lack samples (n=1). "
        "Generator lag exceeded the "
        "configured threshold in the naive run; this is local evidence, not a Groq prediction.",
        message("daily_budget_exhausted")
        + " The API returns 429/503 with Retry-After only when known. "
        "No real daily reset promise is made. Verified cache results remain eligible.",
        "The least certain assumption is real quota accounting: missing cached-token usage leaves "
        "conservative reservations instead of invented zeros. There is no "
        "distributed limiter, so this "
        "is a single-worker service; there is no per-user fairness, so one client "
        "can consume shared "
        "capacity. Coordinated omission means a response-paced generator quietly "
        "sends less traffic "
        "when service slows. Precomputed open-loop arrivals expose that overload "
        "instead of hiding it.",
        "CANDIDATE INPUT REQUIRED: personal feedback on Vaani has not been supplied. No homepage "
        "experience or opinion is attributed to the candidate by the assistant.",
        "CANDIDATE INPUT REQUIRED: full actual conversation exports are missing. "
        "The candidate must identify three prompts, a mistake they caught "
        "and a decision they overruled "
        "from those exports. No personal decisions or human review are fabricated.",
    ]
    feedback = Path("candidate/vaani_feedback.txt")
    if feedback.is_file() and feedback.read_text(encoding="utf-8").strip():
        texts[10] = feedback.read_text(encoding="utf-8").strip()
    # Numeric table cells come directly from aggregate.py. Omit optional load rows only for layout.
    selected = {
        "steady",
        "burst",
        "sustained_overload",
        "minute_limit_halved",
        "daily_exhausted",
        "real_smoke",
    }
    for name in ("Table 2a: Latency", "Table 2b: Outcomes"):
        tables[name]["rows"] = [
            r for r in tables[name]["rows"] if r[0].removesuffix(" (dev)") in selected
        ]
    return list(zip(headings(), texts, strict=True)), tables


def html_table(name: str, data: dict[str, object]) -> str:
    rows = [data["headers"], *data["rows"]]
    return (
        "<b>"
        + html.escape(name)
        + "</b><table>"
        + "".join(
            "<tr>" + "".join("<td>" + html.escape(str(cell)) + "</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def main() -> int:
    parts, tables = report_parts()
    markdown = []
    pages = [[], []]
    insert = {
        1: ["Table 1: Limits found in the real service"],
        2: ["Table 2a: Latency", "Table 2b: Outcomes"],
        5: ["Table 3: Retrieval settings"],
    }
    for index, (heading, text) in enumerate(parts):
        markdown.append("## " + heading + "\n\n" + text)
        page = pages[0 if index < 5 else 1]
        page.append("<h2>" + html.escape(heading) + "</h2><p>" + html.escape(text) + "</p>")
        for name in insert.get(index, []):
            markdown.append(name + "\n\n" + table(tables[name]["headers"], tables[name]["rows"]))
            page.append(html_table(name, tables[name]))
    (ROOT / "writeup.md").write_text("\n\n".join(markdown), encoding="utf-8")
    css = (
        "body{font-family:Arial,sans-serif;font-size:9pt;line-height:1.15} "
        "h2{font-size:10pt;margin:5pt 0 2pt} p{margin:2pt 0 4pt} "
        "table{border-collapse:collapse;font-size:7.3pt;width:100%;margin:3pt 0} "
        "td{border:0.4pt solid #888;padding:1pt 2pt}"
    )
    doc = fitz.open()
    for blocks in pages:
        page = doc.new_page(width=595.28, height=841.89)
        spare, scale = page.insert_htmlbox(
            fitz.Rect(34, 31, 561, 810), "".join(blocks), css=css, scale_low=0.85
        )
        if spare < 0 or scale < 0.85:
            raise ValueError("Report layout does not fit legibly in two pages")
    doc.save(ROOT / "writeup.pdf", deflate=True)
    for index, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(ROOT / f"writeup-page-{index + 1}.png")
    print(f"Rendered {len(doc)}-page report; verify page PNGs before delivery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
