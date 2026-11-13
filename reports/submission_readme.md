# Relay submission

This folder contains the two-page report, source code, results.csv and raw evidence.

| Path | Contents |
|---|---|
| `writeup.pdf` | Design, measurements and limitations |
| `results.csv` | One row per logical request, with the required columns first |
| `code/` | Source, configuration, tests and setup instructions |
| `raw/` | Real provider observations, local events and manifests |
| `evaluation/` | Questions, isolated references and output reviews |
| `reports/` | Audits, website notes and readiness findings |
| `conversation_history/` | Actual supplied prompts; scope described in its README |
| `artifact_manifest.json` | File hashes, sizes and provenance |

From `code/`, run `uv sync --locked --extra dev`, then `uv run --locked --extra dev python -m pytest`. Start the stand-in and Relay in separate terminals using `code/README.md`.

Tables come only from `scripts.aggregate`, the CSV and recorded evidence. Two historical requests reached Groq; the pipeline smoke safely abstained. Real A/B/C quality and complete calibration remain unmeasured. Development runs do not establish Groq capacity. Review records identify the reviewer and distinguish provisional judgments from candidate confirmation. Read `CANDIDATE_ACTIONS.md` before submitting.
