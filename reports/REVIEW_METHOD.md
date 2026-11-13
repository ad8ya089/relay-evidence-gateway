# Output review method

All 54 retained A/B/C outputs were assessed against their questions, pinned references, supplied chunks and citation metadata. A complete answer had to address every material part of the question, rather than quote a true but irrelevant sentence.

The 48 visible scripted responses were marked wrong with individual question-specific rationales. Six timeout outputs had no visible answer and remain uncertain: a transport failure is not evidence of a correct refusal. Denominators are shown in the generated tables. These judgments are provisional assistant reviews; candidate confirmation is not implied.

`supporting_chunk_ids` in the review files identifies the response's cited chunks. `metadata_matches` means the source/version/page fields match those chunks. It does not mean the response fully answers the question. Correct reference passages are supplied in `source_evidence_map.json`.

Five outputs were selected with seed 20261114 and reviewed again using separate output-based rationales. The same assistant performed both passes. The repeated-pass agreement is descriptive and is not an independent human-grader reliability estimate; the sample consists of five wrong answers.

The stand-in is a scripted pipeline exerciser. Its poor answer completeness cannot be used to estimate Groq model quality or to select a retrieval setting for real deployment. Historical failed runs and both review passes remain available.
