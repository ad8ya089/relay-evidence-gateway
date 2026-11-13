"""Persist explicitly authored offline reviews of the retained stand-in outputs."""

import csv
import hashlib
import json
import random
from pathlib import Path

from eval.grade import AssistedGrade, save_assisted_grade

ROOT = Path("results/final_completion")

# These decisions were made by reading actual visible outputs, not by status heuristics.
REASONS = {
    "A01": (
        "Only encoder-decoder attention is described; encoder and masked "
        "decoder self-attention are missing."
    ),
    "A02": (
        "Layer normalization or encoder depth does not explain causal "
        "masking and shifted decoder embeddings."
    ),
    "A03": (
        "Motivates position information but omits sine/cosine construction"
        " and why it was preferred to learned positions."
    ),
    "A04": (
        "Does not give the layer complexities, parallel operations, path "
        "lengths or n<d qualification."
    ),
    "A05": "Does not refuse the unsupported GLUE score request; returns unrelated source text.",
    "A06": (
        "Hardware or translation results do not establish electricity "
        "consumption in kWh; should abstain."
    ),
    "B01": (
        "A dialogue heading or isolated NLU sentence does not explain the "
        "pipeline limitations and full-duplex architecture."
    ),
    "B02": (
        "The selected fragment does not explain semantic distillation and "
        "the parallel seven-level acoustic RVQ."
    ),
    "B03": (
        "A heading or generic streaming fragment does not explain Moshi's "
        "own text stream and its speech scaffolding."
    ),
    "B04": (
        "Does not distinguish 160 ms theoretical and about 200 ms "
        "practical latency; delay training qualification is missing."
    ),
    "B05": (
        "No TED-LIUM 3 WER is established; a codec/metrics heading is not "
        "an appropriate abstention."
    ),
    "B06": (
        "Evaluation table fragments are not a measured maximum "
        "simultaneous-user capacity; should abstain."
    ),
    "C01": (
        "Lists components or discusses Mamba, but omits the requested "
        "information flow and recursive hierarchy."
    ),
    "C02": (
        "A chunking-layer or figure fragment does not explain adjacent "
        "cosine projections and the boundary threshold."
    ),
    "C03": (
        "A citation fragment or voice-instruction heading does not explain"
        " smoothing, gradients and ratio loss."
    ),
    "C04": "Does not supply bits-per-byte, matched bytes per batch or matched FLOPs per byte.",
    "C05": (
        "Does not explicitly abstain from a PDF/RAG chunk-size "
        "recommendation absent from the corpus."
    ),
    "C06": "Model-scale/ablation text does not establish a dollar training bill; should abstain.",
}


def build() -> None:
    output = ROOT / "review"
    output.mkdir(exist_ok=False)
    source = next(Path("results/p7/standin_evaluation_repeat").glob("eval-*/rows.jsonl"))
    expected_source = "2b687a73c64fb223ecc852eb714383e85e5ce2c2167859d96ce87b6b6e34c23c"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source:
        raise ValueError("Authored decisions apply only to the exact reviewed output file")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    chunks = {
        setting: {
            r["chunk_id"]: r
            for line in Path(f"data/chunks/{setting}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            for r in [json.loads(line)]
        }
        for setting in ["A", "B", "C"]
    }
    uncertain = {("A", "B02"), ("C", "A01"), ("C", "A04"), ("C", "B06"), ("C", "C03"), ("C", "C05")}
    queue = []
    for row in rows:
        setting, qid = row["retrieval_setting"], row["question_id"]
        citations = row["citations"]
        valid = all(
            c["chunk_id"] in row["supplied_chunk_ids"]
            and all(
                c[k] == chunks[setting][c["chunk_id"]][k]
                for k in ["arxiv_id", "arxiv_version", "section", "page_start", "page_end"]
            )
            for c in citations
        )
        stage = (
            "retrieval_miss"
            if (setting, qid)
            in {("A", "B01"), ("B", "B01"), ("A", "B03"), ("B", "B03"), ("A", "C04"), ("B", "C04")}
            else "model_misread"
        )
        assessment = dict(
            request_id=row["request_id"],
            question_id=qid,
            retrieval_setting=setting,
            answer_grade="",
            failure_stage="",
            rationale="No visible answer: transport timeout is not a semantic refusal.",
            grader_provenance="assistant_review",
            uncertain=True,
            supporting_chunk_ids=json.dumps([c["chunk_id"] for c in citations]),
            citation_validity="metadata_matches" if citations and valid else "none",
        )
        if (setting, qid) not in uncertain:
            grade = AssistedGrade(
                **{k: v for k, v in assessment.items() if k != "uncertain"}
                | {"answer_grade": "wrong", "failure_stage": stage, "rationale": REASONS[qid]}
            )
            save_assisted_grade(grade, output / "grades.csv")
            assessment.update(grade.model_dump(), uncertain=False)
        queue.append(assessment)
    with (output / "all_reviews.csv").open("w", newline="", encoding="utf-8") as out:
        columns = sorted(set().union(*(q.keys() for q in queue)))
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(queue)
    # Seeded sample reviewed again using the raw question/output, before first-pass comparison.
    second_reasons = {
        (
            "A",
            "A05",
        ): "The visible word 'application' neither supplies a GLUE score nor explains its absence.",
        (
            "A",
            "C04",
        ): (
            "An appendix title about distillation cannot answer the requested "
            "metric and training-budget controls."
        ),
        (
            "B",
            "C06",
        ): (
            "The pool-baseline sentence contains no cloud expenditure and "
            "fails to decline the unsupported request."
        ),
        (
            "C",
            "A02",
        ): "Encoder layer count is unrelated to future-token masking or the decoder input shift.",
        (
            "B",
            "C01",
        ): (
            "The Mamba ablation question omits H-Net's "
            "encode/compress/main/decompress/decode flow and recursion."
        ),
    }
    sample = random.Random(20261114).sample(rows, 5)
    for row in sample:
        grade = AssistedGrade(
            request_id=row["request_id"],
            question_id=row["question_id"],
            retrieval_setting=row["retrieval_setting"],
            answer_grade="wrong",
            round=2,
            failure_stage="model_misread",
            rationale=second_reasons[(row["retrieval_setting"], row["question_id"])],
        )
        save_assisted_grade(grade, output / "regrades.csv")
    (output / "methodology.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "grader_provenance": "assistant_review",
                "human_confirmed": False,
                "rubric": (
                    "Material completeness, support, citation provenance and "
                    "appropriate abstention; see authored rationales."
                ),
                "uncertain": (
                    "Six blank timeout outputs are left ungraded, rather than credited"
                    " as correct refusals."
                ),
                "stage_caveat": (
                    "model_misread denotes generation failure here; the stand-in is "
                    "scripted, not a real reasoning model."
                ),
                "second_pass": (
                    "Same assistant, separate fresh output-based rationales; not an "
                    "independent human grader."
                ),
                "regrade_seed": 20261114,
                "regrade_request_ids": [r["request_id"] for r in sample],
                "candidate_confirmation": "pending",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
