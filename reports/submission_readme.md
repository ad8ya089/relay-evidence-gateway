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

Copy `code/` to a separate working directory before installing dependencies or running tests, so runtime files do not change the delivered manifest. In that working copy, run `uv sync --locked --extra dev`. Corpus integration checks require the pinned paper/model assets: follow the ingestion step in its README before running `uv run --locked --extra dev python -m pytest`. Those asset downloads do not call Groq. Start the stand-in and Relay in separate terminals using the same README.

From that working directory, `uv run --locked --extra dev python -m scripts.check_submission --package <absolute-path-to-relay-submission> --structural-only` recomputes the packaged tables and checks the manifest against the included evidence. Readiness validation intentionally remains blocked where measurements or approvals are missing.

Tables come only from `scripts.aggregate`, the CSV and recorded evidence. Two historical requests reached Groq; the pipeline smoke safely abstained. Real A/B/C quality and complete calibration remain unmeasured. Development runs do not establish Groq capacity. Review records identify the reviewer and distinguish provisional judgments from candidate confirmation. Read `CANDIDATE_ACTIONS.md` before submitting.
