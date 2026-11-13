"""Ground literal candidate reference phrases in pinned chunks, outside model execution."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from eval.dataset import load_key
from relay.retrieval.models import Chunk


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def associate(data: Path = Path("data")) -> dict[str, object]:
    chunks = [
        Chunk.model_validate_json(line)
        for line in (data / "chunks/A.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    result = {}
    for qid, key in load_key().items():
        candidates = []
        if key.answerable:
            target = words(key.gold_reference)
            for chunk in chunks:
                if (chunk.paper_id, chunk.arxiv_id, chunk.arxiv_version) != (
                    key.paper,
                    key.arxiv_id,
                    key.arxiv_version,
                ):
                    continue
                source = words(chunk.text)
                match = SequenceMatcher(None, target, source, autojunk=False).find_longest_match()
                if match.size >= 6:
                    positions = list(re.finditer(r"\w+", chunk.text))
                    quote = chunk.text[
                        positions[match.b].start() : positions[match.b + match.size - 1].end()
                    ]
                    candidates.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "paper": chunk.paper_id,
                            "arxiv_id": chunk.arxiv_id,
                            "arxiv_version": chunk.arxiv_version,
                            "section": chunk.section,
                            "page_start": chunk.page_start,
                            "phrase": " ".join(source[match.b : match.b + match.size]),
                            "quote": quote,
                            "words": match.size,
                        }
                    )
        candidates.sort(key=lambda row: (-row["words"], row["chunk_id"]))
        result[qid] = {
            "evidence_hint": key.evidence_hint,
            "anchors": candidates[:2],
            "basis": "literal overlap with candidate answer notes in actual pinned chunks",
            "human_support_review": "pending",
            "answerable": key.answerable,
        }
    return result


def hit(texts: list[str], anchors: list[dict[str, object]]) -> bool | None:
    if not anchors:
        return None
    return any(
        str(anchor["phrase"]) in " ".join(words(text)) for text in texts for anchor in anchors
    )


def validate_anchors(chunks: dict[str, Chunk], anchors: list[dict[str, object]]) -> None:
    for anchor in anchors:
        chunk = chunks.get(str(anchor["chunk_id"]))
        if chunk is None or not str(anchor["quote"]) or str(anchor["quote"]) not in chunk.text:
            raise ValueError("Reference quote is not grounded in an actual chunk")
        if (
            anchor["arxiv_id"],
            anchor["arxiv_version"],
            anchor["section"],
            anchor["page_start"],
        ) != (chunk.arxiv_id, chunk.arxiv_version, chunk.section, chunk.page_start):
            raise ValueError("Reference source/version/page metadata mismatch")


if __name__ == "__main__":
    path = Path("results/p7/reference_associations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(associate(), indent=2), encoding="utf-8")
    print("Grounded reference associations saved; human evidence review remains pending")
