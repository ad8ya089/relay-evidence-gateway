import hashlib
from pathlib import Path

import httpx
import pymupdf
import pytest

from relay.retrieval.ingest import IngestionError, fetch_paper, file_hash, ingest, verify_pdf
from relay.settings import retrieval_settings


def fixture_pdf(path: Path, stamp: str) -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), stamp)
        page.insert_text((50, 80), "1 FAKE Methods", fontsize=14)
        page.insert_text((50, 120), "FAKE source evidence only for offline integration tests.")
        pdf.save(path)
    return path.read_bytes()


@pytest.mark.parametrize("version,valid", [(7, True), (6, False), (70, False)])
def test_exact_version_stamp(tmp_path: Path, version: int, valid: bool) -> None:
    paper = retrieval_settings().papers[0]
    path = tmp_path / "fixture.pdf"
    fixture_pdf(path, f"arXiv:1706.03762v{version}")
    if valid:
        assert verify_pdf(path, paper) == 1
    else:
        with pytest.raises(IngestionError, match="expected arXiv"):
            verify_pdf(path, paper)


@pytest.mark.parametrize("stamp", ["no stamp", "arXiv:2410.00037v7"])
def test_missing_or_wrong_id_rejected(tmp_path: Path, stamp: str) -> None:
    path = tmp_path / "fixture.pdf"
    fixture_pdf(path, stamp)
    with pytest.raises(IngestionError):
        verify_pdf(path, retrieval_settings().papers[0])


def test_invalid_pdf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(IngestionError, match="signature"):
        verify_pdf(path, retrieval_settings().papers[0])


async def test_verified_download_is_idempotent(tmp_path: Path) -> None:
    paper = retrieval_settings().papers[0]
    content = fixture_pdf(tmp_path / "source.pdf", "arXiv:1706.03762v7")
    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        first = await fetch_paper(paper, tmp_path / "papers", client)
        second = await fetch_paper(paper, tmp_path / "papers", client, first)
    assert first == second
    assert calls == [paper.url]
    assert first.sha256 == file_hash(tmp_path / "papers/transformer-v7.pdf")
    assert first.fetched_at.endswith("+00:00")


@pytest.mark.parametrize("status,body", [(404, b"missing"), (200, b"<html>oops</html>")])
async def test_http_or_invalid_response_aborts(tmp_path: Path, status: int, body: bytes) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, content=body))
    ) as client:
        with pytest.raises((httpx.HTTPStatusError, IngestionError)):
            await fetch_paper(retrieval_settings().papers[0], tmp_path / "papers", client)
    assert not list((tmp_path / "papers").glob("*.pdf"))


async def test_cached_hash_mismatch_aborts(tmp_path: Path) -> None:
    paper = retrieval_settings().papers[0]
    content = fixture_pdf(tmp_path / "source.pdf", "arXiv:1706.03762v7")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=content))
    ) as client:
        doc = await fetch_paper(paper, tmp_path / "papers", client)
        path = tmp_path / "papers/transformer-v7.pdf"
        path.write_bytes(content + b"\n% changed")
        with pytest.raises(IngestionError, match="hash differs"):
            await fetch_paper(paper, tmp_path / "papers", client, doc)


async def test_full_ingestion_and_indexes_are_offline_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = retrieval_settings().papers
    content = {
        p.url: fixture_pdf(tmp_path / f"{p.paper_id}.pdf", f"arXiv:{p.arxiv_id}v{p.arxiv_version}")
        for p in papers
    }
    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=content[str(request.url)])

    def transport(**kwargs: object) -> httpx.MockTransport:
        assert kwargs["retries"] == 0
        return httpx.MockTransport(serve)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    root = tmp_path / "data"
    first = await ingest(root)
    second = await ingest(root)
    assert calls == [p.url for p in papers]
    assert first == second and len(first.documents) == 3
    assert (
        first.corpus_hash
        == hashlib.sha256("".join(d.sha256 for d in first.documents).encode()).hexdigest()
    )
    assert (root / "chunks/C.jsonl").read_bytes() == (root / "chunks/D.jsonl").read_bytes()
    assert all((root / f"indexes/{name}.lexical.json").exists() for name in "ABCD")
    assert (root / "extraction_diagnostics.json").exists()
    assert not (root / ".ingest-progress.json").exists()


async def test_failed_ingest_never_writes_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = retrieval_settings().papers[0]
    content = fixture_pdf(tmp_path / "source.pdf", f"arXiv:{paper.arxiv_id}v{paper.arxiv_version}")
    calls: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(200, content=content)
        return httpx.Response(503)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(serve))
    root = tmp_path / "data"
    with pytest.raises(httpx.HTTPStatusError):
        await ingest(root)
    assert not (root / "corpus_manifest.json").exists()
    assert (root / ".ingest-progress.json").exists()
