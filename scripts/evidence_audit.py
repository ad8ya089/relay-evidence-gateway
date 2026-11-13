"Offline review aids, provenance and budget diagnosis; never human grades or real calls."

import csv
import json
from pathlib import Path

import fitz

from eval.dataset import load_key
from relay.app.prompts import Prompts
from relay.control.settings import load_policy
from relay.retrieval.models import RetrievalHit
from relay.retrieval.tokenizer import get_encoding
from relay.settings import SafetySettings

ROOT = Path("results/evidence_completion")


def build() -> None:
    pages = {
        "A01": [5],
        "A02": [3, 5],
        "A03": [6],
        "A04": [6, 7],
        "B01": [1, 6, 7, 15],
        "B02": [10, 11, 12],
        "B03": [15, 16],
        "B04": [1, 26],
        "C01": [4, 5],
        "C02": [6],
        "C03": [7, 8],
        "C04": [11],
    }
    pdfs = {
        paper: fitz.open(next(Path("data/papers").glob(paper + "-*.pdf")))
        for paper in ["transformer", "moshi", "dynamic_chunking"]
    }
    review = []
    for key in load_key().values():
        locations = []
        for page in pages.get(key.qid, []):
            text = pdfs[key.paper][page - 1].get_text()
            dest = ROOT / "reference_pages" / f"{key.paper}-p{page}.txt"
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            locations.append(
                {"pdf_page": page, "text_artifact": str(dest), "section_hint": key.evidence_hint}
            )
        cautions = {
            "A02": (
                "Check the distinction between current decoder position and shifted "
                "output embeddings."
            ),
            "A04": (
                "Verify table equations visually: PDF text extraction separates "
                "exponents; retain n<d condition."
            ),
            "B02": (
                "512 is pre-quantization latent dimension; PDF p11 also describes "
                "projection to 256 before RVQ. Do not imply every internal vector is "
                "512-dimensional."
            ),
            "B03": (
                "Text is Moshi's own speech, not the user stream. "
                "Alignment details continue on p16."
            ),
            "B04": (
                "160 ms is theoretical, about 200 ms practical. Pretraining delay 2 / "
                "finetuning delay 1 supported on p26."
            ),
            "C02": (
                "Verify cosine equation visually on p6; first boundary p1=1.0 and threshold >=0.5."
            ),
            "C03": (
                "F=G=1/N is the equal-F/G minimum condition; the paper explicitly "
                "permits loss below 1 when F differs from G. Do not claim universal "
                "minimizer or exact enforced compression."
            ),
        }
        review.append(
            {
                "question_id": key.qid,
                "human_review_complete": False,
                "candidate_claims_to_review": key.gold_reference,
                "locations": locations,
                "assistant_assessment": (
                    "Main claims supported in listed pinned passages; candidate sign-off pending"
                )
                if key.answerable
                else (
                    "Requested fact not established in pinned corpus review; absence is "
                    "bounded to these three versions"
                ),
                "questionable_or_qualified": cautions.get(
                    key.qid,
                    "No specific correction identified; "
                    "candidate must check all claims and citations",
                ),
                "literal_phrase_anchors": key.evidence_anchors,
                "literal_anchor_status": "present"
                if key.evidence_anchors
                else "missing; candidate should supply exact evidence phrases"
                if key.answerable
                else "not applicable: unanswerable",
            }
        )
    (ROOT / "reference_review.json").write_text(json.dumps(review, indent=2))
    diagnosis = [
        "# Offline smoke diagnosis â€” evidence incomplete",
        "Recorded: quote_not_substring for claim 0; membership and provenance passed.",
        "Outcome has digests, not rejected draft/quote strings. Provider body_trunc is null.",
        "Stream chunks are empty. The ledger stores usage, not draft content.",
        "The one-off driver did not persist LLMResult.text. Digests cannot recover input.",
        "Exact characters, Unicode differences and quote-to-chunk mapping are unavailable.",
        "Paraphrase, incorrect chunk association and extraction damage cannot be distinguished.",
        "No validator defect is demonstrated. Prompts are unchanged.",
        "Six retrieved IDs are saved; transformer-v7-p5-c20 is section 3.2.3, PDF p5.",
        "Literal spans pass the boundary test. Altered text and unknown IDs fail.",
        "This is not a reconstruction of the missing failed draft.",
        "Sealed abstention is preserved. No verifier/repair ran. No new real smoke is authorized.",
    ]
    (ROOT / "smoke_diagnosis.md").write_text("\n\n".join(diagnosis), encoding="utf-8")
    attempts = json.loads(Path("results/p7/real_smoke/provider_calls.json").read_text())["attempts"]
    calibration = {
        "status": "partial; not a complete calibrated profile",
        "verifier_samples": 0,
        "source": "historical actual Groq smoke",
        "observations": [],
    }
    for a in attempts:
        r = a["result"]
        calibration["observations"].append(
            {
                "provider_request_id": r["provider_request_id"],
                "kind": a["kind"],
                "sample_count": 1,
                "ttft_ms": r["provider_ttft_ms"],
                "duration_ms": r["total_ms"],
                "usage": r["usage"],
                "method": "raw single observation; no distribution fit or percentile inference",
                "uncertainty": "one draft sample; no verifier or complete streaming decode series",
            }
        )
    (ROOT / "calibration_manifest.json").write_text(json.dumps(calibration, indent=2))
    policy, safety = load_policy(), SafetySettings()
    plans = []
    for setting, e in policy.estimates.items():
        d = e.draft_input + policy.normal.draft_max_completion_tokens
        v = e.verify_input + policy.normal.verify_max_completion_tokens
        # Structural call count: draft/verify + repair/verify, plus request-wide retries.
        originals = 2 * (1 + policy.normal.repair_max)
        plans.append(
            {
                "setting": setting,
                "configured_initial_pair_estimate": d + v + policy.safety_tokens,
                "original_calls_max": originals,
                "retry_calls_max_request_wide": policy.retry.max_per_request,
                "provider_calls_max": originals + policy.retry.max_per_request,
                "conditional_18_question_estimate": 18
                * (
                    (d + v) * (1 + policy.normal.repair_max)
                    + max(d, v) * policy.retry.max_per_request
                    + policy.safety_tokens
                ),
                "estimate_caveat": (
                    "Configured input estimates, not strict input bounds. Repair adds "
                    "rejected draft/feedback; actual messages must be tokenized before "
                    "dispatch. No invented p95 from one draft."
                ),
                "observed_complete_logical_usage": None,
                "observed_draft_usage": 1974 if setting == "A" else None,
            }
        )
    budget = {
        "status": "blocked; fresh quota screenshot unavailable",
        "new_real_calls": 0,
        "previous_formula_defect": (
            "Multiplied all draft/verifier pairs by retry factor; "
            "retry.max_per_request is global to the logical request, not per call. "
            "Repair prompt growth was omitted."
        ),
        "plans": plans,
        "conditional_54_estimate": sum(p["conditional_18_question_estimate"] for p in plans),
        "session_token_reservation_cap": safety.real_token_cap,
        "session_request_cap": safety.real_request_cap,
        "absolute_usage_caveat": (
            "Guard enforces estimated reservations before dispatch, reconciles "
            "reported usage afterwards. Tokenizer estimation is not a provider-"
            "guaranteed absolute usage bound."
        ),
        "fresh_quota_evidence": "not supplied; historical headers cannot authorize new dispatch",
        "high_percentile": "not measured",
        "standalone_smoke_authorized": False,
    }
    (ROOT / "budget_review.json").write_text(json.dumps(budget, indent=2))
    questions = {
        json.loads(line)["qid"]: json.loads(line)["question"]
        for line in Path("eval/datasets/candidate-v1/questions.jsonl").read_text().splitlines()
    }
    prompts = []
    for line in Path("results/p7/retrieval_observations.jsonl").read_text().splitlines():
        observation = json.loads(line)
        hits = [RetrievalHit.model_validate(h) for h in observation["hits"]]
        messages = Prompts().draft(questions[observation["question_id"]], hits)
        framed = json.dumps(
            [m.model_dump() for m in messages], ensure_ascii=False, separators=(",", ":")
        )
        prompts.append(
            {
                "question_id": observation["question_id"],
                "setting": observation["setting"],
                "context_tokens": sum(h.n_tokens for h in hits),
                "serialized_draft_input_tokens": len(get_encoding().encode(framed)),
                "completion_cap": policy.normal.draft_max_completion_tokens,
                "source": "offline reconstruction from saved retrieval hits; not Groq usage",
            }
        )
    (ROOT / "prompt_token_audit.json").write_text(json.dumps(prompts, indent=2))
    rows_path = next(Path("results/p7/standin_evaluation_repeat").glob("eval-*/rows.jsonl"))
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    with (ROOT / "grading_queue.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "request_id",
                "question_id",
                "setting",
                "answerable",
                "answer_grade",
                "failure_stage",
                "rationale",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "request_id": r["request_id"],
                    "question_id": r["question_id"],
                    "setting": r["retrieval_setting"],
                    "answerable": load_key()[r["question_id"]].answerable,
                }
            )
    (ROOT / "grading_manifest.json").write_text(
        json.dumps(
            {
                "source": "stand-in; ungraded",
                "rows_file": str(rows_path),
                "awaiting_review": len(rows),
                "per_setting": 18,
                "answerable_per_setting": 12,
                "unanswerable_per_setting": 6,
                "missing_grades": len(rows),
                "regrade": (
                    "After initial human grading, use eval.grade --regrade 5 --round 2 "
                    "--seed 20261114; report agreement"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
