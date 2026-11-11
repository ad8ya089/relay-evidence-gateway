"""Explicit-network, exact-version ingestion CLI. Never invoked by imports/tests."""

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pymupdf
from pydantic import BaseModel

from relay.retrieval.chunking import build_chunks, extract_pdf, extraction_summary
from relay.retrieval.models import CorpusManifest, DocumentManifest
from relay.settings import PaperSpec, retrieval_settings


class IngestionError(ValueError):
    """Invalid PDF, unverified cached content, or exact version mismatch."""


class IngestionProgress(BaseModel):
    documents: list[DocumentManifest]


def verify_pdf(path: Path, paper: PaperSpec) -> int:
    if not path.read_bytes().startswith(b"%PDF-"):
        raise IngestionError(f"{paper.paper_id}: invalid PDF signature")
    try:
        with pymupdf.open(path) as pdf:
            if not pdf.page_count:
                raise IngestionError(f"{paper.paper_id}: empty PDF")
            stamps = re.findall(r"arXiv:\s*(\d{4}\.\d{4,5})\s*v(\d+)\b", pdf[0].get_text())
            expected = (paper.arxiv_id, str(paper.arxiv_version))
            if not stamps or set(stamps) != {expected}:
                raise IngestionError(
                    f"{paper.paper_id}: expected arXiv:{paper.arxiv_id}v{paper.arxiv_version}; "
                    f"page-one stamps were {stamps!r}"
                )
            return pdf.page_count
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise IngestionError(f"{paper.paper_id}: unreadable PDF") from exc


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def fetch_paper(
    paper: PaperSpec,
    directory: Path,
    client: httpx.AsyncClient,
    previous: DocumentManifest | None = None,
) -> DocumentManifest:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{paper.paper_id}-v{paper.arxiv_version}.pdf"
    if path.exists():
        pages = verify_pdf(path, paper)
        sha = file_hash(path)
        if previous is not None:
            if sha != previous.sha256:
                raise IngestionError(f"{paper.paper_id}: cached hash differs from manifest")
            if any(getattr(previous, key) != value for key, value in paper.model_dump().items()):
                raise IngestionError(f"{paper.paper_id}: cached manifest metadata mismatch")
            return previous
        # Only files with recorded fetch timestamps qualify as verified cache entries.
        # A valid orphan (e.g. interrupted prior run) is fetched again to record provenance.
    response = await client.get(paper.url)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-"):
        raise IngestionError(f"{paper.paper_id}: response is not a PDF")
    temporary = path.with_suffix(".pdf.part")
    try:
        temporary.write_bytes(response.content)
        pages = verify_pdf(temporary, paper)
        sha = file_hash(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return DocumentManifest(
        **paper.model_dump(),
        sha256=sha,
        page_count=pages,
        # This is required provenance, never used as a duration clock.
        fetched_at=datetime.now(UTC).isoformat(),
    )


async def ingest(root: Path = Path("data"), *, dense: bool = False) -> CorpusManifest:
    config = retrieval_settings()
    manifest_path = root / "corpus_manifest.json"
    progress_path = root / ".ingest-progress.json"
    previous = (
        CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    prior_docs = {d.paper_id: d for d in previous.documents} if previous else {}
    if progress_path.exists():
        progress = IngestionProgress.model_validate_json(progress_path.read_text(encoding="utf-8"))
        prior_docs.update({d.paper_id: d for d in progress.documents})
    documents: list[DocumentManifest] = []
    async with httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=0),
        follow_redirects=True,
        timeout=config.download_timeout_s,
    ) as client:
        for paper in config.papers:
            doc = await fetch_paper(paper, root / "papers", client, prior_docs.get(paper.paper_id))
            documents.append(doc)
            # Persist verified downloads incrementally, allowing interrupted runs to resume.
            progress_path.write_text(
                IngestionProgress(documents=documents).model_dump_json(indent=2), encoding="utf-8"
            )
            print(
                f"Verified {doc.paper_id} arXiv:{doc.arxiv_id}v{doc.arxiv_version}, "
                f"{doc.page_count} pages, sha256={doc.sha256}"
            )
    manifest = CorpusManifest(
        documents=documents,
        corpus_hash=hashlib.sha256("".join(d.sha256 for d in documents).encode()).hexdigest(),
    )
    temporary_manifest = manifest_path.with_suffix(".json.part")
    temporary_manifest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    progress_path.unlink(missing_ok=True)
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    extracts = {
        paper.paper_id: extract_pdf(
            root / "papers" / f"{paper.paper_id}-v{paper.arxiv_version}.pdf", config
        )
        for paper in config.papers
    }
    for name in ("A", "B", "C"):
        chunks = [
            chunk
            for paper in config.papers
            for chunk in build_chunks(
                extracts[paper.paper_id], paper, config.settings[name], config
            )
        ]
        (chunks_dir / f"{name}.jsonl").write_text(
            "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8"
        )
        print(f"{name}: {len(chunks)} chunks")
    (chunks_dir / "D.jsonl").write_bytes((chunks_dir / "C.jsonl").read_bytes())
    (root / "extraction_diagnostics.json").write_text(
        json.dumps({key: extraction_summary(value) for key, value in extracts.items()}, indent=2),
        encoding="utf-8",
    )
    # Persist lexical indexes; dense embeddings are deliberately lazy and separately cached.
    from relay.retrieval.search import RetrievalEngine

    engine = RetrievalEngine(data_dir=root)
    for name in config.settings:
        engine.prepare_lexical(name)
    if dense:
        print("Preparing the C/D dense index outside the per-query deadline...", flush=True)
        count = await asyncio.to_thread(engine.prepare_dense, "C")
        print(f"C/D: dense index prepared for {count} chunks", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense", action="store_true", help="explicitly build the C/D vector index"
    )
    args = parser.parse_args()
    try:
        asyncio.run(ingest(dense=args.dense))
    except (IngestionError, httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        print(f"Ingestion aborted: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
