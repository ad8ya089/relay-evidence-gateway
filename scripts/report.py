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
        "I built Relay as an application-owned pipeline: interpret, retrieve "
        "through a tool, check evidence, "
        "draft, validate, verify and release. My main choice was to buffer the "
        "answer until validation. "
        "That makes the first visible content slower, but avoids streaming "
        "unsupported claims. Admission "
        "and quotas decide whether work can finish inside the deadline.",
        "The compatible adapter talks to either Groq or the local stand-in. Two"
        " historical real attempts "
        "are retained: connectivity and a smoke that safely abstained after "
        "quote validation failed. "
        "The rejected quote was not saved, so its exact defect cannot be "
        "reconstructed. Calibration has "
        "one draft TTFT sample and no verifier sample. Development latency, "
        "concurrency and fault settings "
        "remain assumptions; they do not establish Groq capacity.",
        "Percentiles use NumPy linear interpolation over status=ok, with n "
        "shown. Failures are counted "
        "separately; first token means user-visible content. Rows marked (dev) "
        "use an uncalibrated stand-in. "
        "The halved-quota repeat now binds. All earlier failures remain in CSV."
        " The output audit found "
        f"{metrics['review_summary']['reviewed']} materially incomplete "
        "or unrelated stand-in answers; "
        f"{metrics['review_summary']['uncertain']} blank timeout outputs remain uncertain. "
        "A valid citation was not enough to answer the question. These are "
        "assistant reviews, not human "
        "grades or Groq quality measurements. Unknown usage and prices remain not measured.",
        f"P6 development request {timeline['request_id']}: "
        f"waiting={timeline['waiting_ms']} ms; "
        f"working={timeline['working_ms']} ms; retrying={timeline['retrying_ms']} ms; "
        f"unattributed={timeline['unknown_ms']} ms. First visible content was "
        f"{timeline['user_visible_first_token_ms']} ms; "
        f"after-first={timeline['after_first_token_ms']} ms. "
        "The delay was before release. This split comes from raw event "
        "intervals and client timestamps; "
        "it is a stand-in trace, not a real-provider latency claim.",
        "Admission delays work only when the projected wait fits the deadline. "
        "Otherwise it rejects "
        "without a model call. Pressure mode reduces retrieved context and "
        "output caps, retaining validation. "
        "In the local minute-quota run, "
        f"{metrics['minute_limit_halved']['outcomes']['rate_limited']:.1f}% "
        "were rate-limited after the limit change. Daily exhaustion stops new "
        "work but permits verified "
        "cache hits. No real recovery time or reset rule has been established.",
        "A/B use lexical retrieval; C uses hybrid retrieval and attempts "
        "reranking, recording fallback. "
        "Each condition asked the same 18 questions. Reviewed stand-in outputs did not establish a "
        "quality winner, so I have not selected a setting on that basis. Six "
        "timed-out outputs remain "
        "ungraded. Reference corrections and review rationales are supplied separately. Gold never "
        "enters model requests; real retrieval quality remains unmeasured.",
        "B01's conversational wording ranked a misspelling dialogue first "
        "(moshi-v1-p64-c325, F.2), "
        "rather than architecture. B02's meaning/acoustic wording ranked an entropy dialogue first "
        "(moshi-v1-p62-c319, F.1), rather than Mimi's split RVQ. C03 ranked an overview "
        "(dynamic_chunking-v2-p2-c7), rather than smoothing and ratio loss. "
        "Supporting passages and "
        "full retrieved sets are retained; these top-one mistakes do not prove a top-k miss.",
        "Only the orchestrator retries, with a request-wide cap, jitter, "
        "deadlines and a global budget. "
        "The repeated local pair kept workload, seed, schedule, profile, faults and limits equal. "
        f"Naive success was {metrics['retry_naive']['outcomes']['ok']:.1f}% versus "
        f"bounded {metrics['retry_bounded']['outcomes']['ok']:.1f}% "
        f"({metrics['retry_naive']['sent']} requests each); provider calls were "
        f"{metrics['retry_naive']['provider_calls_total']} versus "
        f"{metrics['retry_bounded']['provider_calls_total']}. "
        "Naive worsened completion, not provider-call "
        "count: slow work spawned duplicates. Both repeat generator-lag checks passed. "
        "Naive successful percentiles lack samples; this is not a Groq prediction.",
        message("daily_budget_exhausted")
        + " The API returns 429/503, with Retry-After only when known. "
        "Verified cached answers remain eligible; no exact reset time is promised.",
        "My least certain assumption is real quota accounting: missing cached-token usage leaves "
        "conservative reservations. If provider accounting differs, Relay may "
        "shed too early or encounter "
        "a real quota error. I left out a distributed limiter and per-user "
        "fairness, so this is one worker "
        "and one client can consume shared capacity. The useful lesson was coordinated omission: a "
        "response-paced generator hides overload by slowing its own arrivals. "
        "Open-loop schedules reveal it.",
        "Draft for candidate approval: "
        + Path("reports/vaani_feedback_draft.md").read_text().strip(),
        "The evidence-pipeline, control-plane and final-review prompts mattered"
        " most: they fixed the "
        "release boundary, made quota costs explicit and required traceable "
        "measurements. I supplied the "
        "18 question drafts. After reading the earlier PDF, I flagged missing "
        "console-limit provenance, "
        "the unaccounted error outcome and the missing retry experiment. I "
        "overruled omitting the console "
        "values by requiring a separate provenance label. Implementation and "
        "preliminary reviews used "
        "the assistant; raw records identify the reviewer. Actual supplied prompts are attached. "
        "Full exports and final candidate approval remain pending.",
    ]
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
