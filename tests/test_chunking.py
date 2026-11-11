from pathlib import Path

import pymupdf
import pytest

from relay.retrieval.chunking import (
    Extraction,
    Passage,
    build_chunks,
    extract_pdf,
    paragraph_windows,
    token_ids,
)
from relay.retrieval.tokenizer import get_encoding
from relay.settings import Setting, retrieval_settings


@pytest.mark.parametrize("setting", ["A", "B", "C", "D"])
def test_chunk_provenance_sizes_overlap_and_partial_tail(setting: Setting) -> None:
    config = retrieval_settings()
    cfg = config.settings[setting]
    encoding = get_encoding()
    prose = " ".join(f"FAKE paragraph word{i}" for i in range(1000))
    extraction = Extraction(
        passages=[
            Passage(page=1, section="1 FAKE section", kind="text", text=prose, bbox=(0, 0, 1, 1)),
            Passage(
                page=2, section="2 FAKE section", kind="text", text="Short tail.", bbox=(0, 0, 1, 1)
            ),
        ],
        diagnostics=[],
    )
    chunks = build_chunks(extraction, config.papers[0], cfg, config)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.arxiv_id == "1706.03762" and chunk.arxiv_version == 7
        assert chunk.paper_id == "transformer" and chunk.title and chunk.section
        assert 1 <= chunk.page_start <= chunk.page_end <= 2
        assert chunk.n_tokens == len(token_ids(chunk.text, encoding))
        assert chunk.n_tokens <= cfg.target_tokens * (1 + config.chunk_tolerance)
        assert chunk.chunk_id.startswith(f"transformer-v7-p{chunk.page_start}-c")
    page_one = [c for c in chunks if c.page_start == 1]
    for left, right in zip(page_one, page_one[1:], strict=False):
        assert cfg.target_tokens * (1 - config.chunk_tolerance) <= left.n_tokens
        assert (
            token_ids(left.text, encoding)[-cfg.overlap :]
            == token_ids(right.text, encoding)[: cfg.overlap]
        )
    assert chunks[-1].text == "Short tail."
    assert chunks[-1].n_tokens < cfg.target_tokens * (1 - config.chunk_tolerance)


def test_paragraph_boundary_preferred() -> None:
    encoding = get_encoding()
    cfg = retrieval_settings().settings["A"]
    paragraphs = ["FAKE paragraph. " * 28, "FAKE following. " * 60]
    windows = paragraph_windows(
        paragraphs, cfg, encoding, tolerance=retrieval_settings().chunk_tolerance
    )
    assert len(windows) >= 2
    assert all(len(token_ids(w, encoding)) <= cfg.target_tokens for w in windows)
    assert "FAKE" in windows[-1]


def test_columns_section_paths_and_special_kinds(tmp_path: Path) -> None:
    path = tmp_path / "columns.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 40), "1 FAKE Methods", fontsize=14)
        page.insert_text((40, 80), "LEFT first paragraph", fontsize=10)
        page.insert_text((40, 150), "LEFT second paragraph", fontsize=10)
        page.insert_text((320, 80), "RIGHT first paragraph", fontsize=10)
        page.insert_text((320, 150), "RIGHT second paragraph", fontsize=10)
        page = pdf.new_page()
        page.insert_text((40, 40), "1.1 FAKE Detail", fontsize=14)
        page.insert_text((40, 80), "Table 1: FAKE table caption", fontsize=10)
        page.insert_text((40, 130), "References", fontsize=14)
        page.insert_text((40, 170), "FAKE reference text", fontsize=10)
        page = pdf.new_page()
        page.insert_text((40, 40), "Appendix", fontsize=14)
        page.insert_text((40, 80), "FAKE appendix material", fontsize=10)
        pdf.save(path)
    extracted = extract_pdf(path, retrieval_settings())
    page_one = [p.text for p in extracted.passages if p.page == 1]
    assert page_one.index("LEFT second paragraph") < page_one.index("RIGHT first paragraph")
    assert extracted.diagnostics[0].columns == 2
    detail = next(p for p in extracted.passages if "Detail" in p.text)
    assert detail.section == "1 FAKE Methods > 1.1 FAKE Detail"
    assert {p.kind for p in extracted.passages} >= {"text", "caption", "references", "appendix"}


def test_table_kind_and_short_prose_copy() -> None:
    config = retrieval_settings()
    extracted = Extraction(
        passages=[
            Passage(page=1, section="FAKE", kind="table", text="FAKE A 1 B 2", bbox=(0, 0, 1, 1))
        ],
        diagnostics=[],
    )
    chunks = build_chunks(extracted, config.papers[0], config.settings["C"], config)
    assert {c.kind for c in chunks} == {"table", "text"}
    assert len({c.chunk_id for c in chunks}) == 2


def test_borderless_table_cells_remain_table_kind(tmp_path: Path) -> None:
    path = tmp_path / "table.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 45), "Table 1: FAKE fixture cells")
        for row, label in enumerate(("FAKE header", "FAKE alpha", "FAKE beta", "FAKE gamma")):
            for x, text in ((50, label), (240, f"value {row}"), (400, f"result {row}")):
                page.insert_text((x, 80 + row * 25), text, fontsize=10)
        page.insert_text((50, 220), "1 FAKE Following", fontsize=14)
        page.insert_text((50, 260), "FAKE following prose must remain text", fontsize=10)
        pdf.save(path)
    extracted = extract_pdf(path, retrieval_settings())
    assert any(p.kind == "table" and "FAKE alpha" in p.text for p in extracted.passages)
    assert next(p for p in extracted.passages if "following prose" in p.text).kind == "text"


def test_short_caption_also_kept_in_surrounding_prose() -> None:
    config = retrieval_settings()
    extracted = Extraction(
        passages=[
            Passage(
                page=1,
                section="FAKE",
                kind="text",
                text="FAKE surrounding prose.",
                bbox=(0, 0, 1, 1),
            ),
            Passage(
                page=1, section="FAKE", kind="caption", text="Figure 1 FAKE", bbox=(0, 1, 1, 2)
            ),
        ],
        diagnostics=[],
    )
    chunks = build_chunks(extracted, config.papers[0], config.settings["C"], config)
    text = next(c for c in chunks if c.kind == "text")
    assert "FAKE surrounding prose" in text.text and "Figure 1 FAKE" in text.text
    assert any(c.kind == "caption" for c in chunks)


def test_unicode_windows_never_invent_replacement_characters() -> None:
    config = retrieval_settings()
    text = "अक्षर 中文 café 😀 " * 300
    windows = paragraph_windows(
        [text], config.settings["A"], get_encoding(), tolerance=config.chunk_tolerance
    )
    assert all("\ufffd" not in w for w in windows)
