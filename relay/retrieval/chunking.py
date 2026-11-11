"""Block-order PDF extraction and provenance-preserving paragraph windows.

Page references are one-based PDF pages. o200k_base is an approximation of
provider token usage, not a claim about Groq's tokenizer.
"""

import re
from collections import Counter
from pathlib import Path
from statistics import median

import pymupdf
import tiktoken
from pydantic import BaseModel

from relay.retrieval.models import Chunk, ChunkKind
from relay.retrieval.tokenizer import get_encoding
from relay.settings import ChunkSetting, PaperSpec, RetrievalSettings

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+([A-Z].{1,110})$")
CAPTION = re.compile(r"^(Table|Figure|Fig\.)\s+\d+", re.I)
STAMP = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}v\d+", re.I)


class Passage(BaseModel):
    page: int
    section: str | None
    kind: ChunkKind
    text: str
    bbox: tuple[float, float, float, float]


class PageDiagnostic(BaseModel):
    page: int
    blocks: int
    columns: int
    tables_detected: int
    warnings: list[str]


class Extraction(BaseModel):
    passages: list[Passage]
    diagnostics: list[PageDiagnostic]


def token_ids(text: str, encoding: tiktoken.Encoding) -> list[int]:
    return encoding.encode(text, disallowed_special=())


def _reading_order(
    blocks: list[Passage], width: float, config: RetrievalSettings
) -> tuple[list[Passage], int]:
    """Order left/right columns between spanning blocks (title, wide table).

    A page is two-column only if multiple narrow blocks occur on both sides;
    this avoids treating indented prose on single-column pages as a column.
    """
    midpoint = width / 2
    left = [b for b in blocks if b.bbox[2] <= midpoint + width * config.column_slack_ratio]
    right = [b for b in blocks if b.bbox[0] >= midpoint - width * config.column_slack_ratio]
    # A tiny centred block can fit both tests; assign it once, by centre.
    shared = [b for b in left if b in right]
    left = [b for b in left if b not in shared or (b.bbox[0] + b.bbox[2]) / 2 < midpoint]
    right = [b for b in right if b not in shared or (b.bbox[0] + b.bbox[2]) / 2 >= midpoint]
    if len(left) < config.min_column_blocks or len(right) < config.min_column_blocks:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])), 1
    spanning = sorted(
        [b for b in blocks if b not in left and b not in right], key=lambda b: b.bbox[1]
    )
    remaining = [b for b in blocks if b not in spanning]
    ordered: list[Passage] = []
    for span in spanning:
        above = [b for b in remaining if b.bbox[1] < span.bbox[1]]
        for column in (left, right):
            ordered.extend(sorted([b for b in above if b in column], key=lambda b: b.bbox[1]))
        ordered.append(span)
        remaining = [b for b in remaining if b not in above]
    for column in (left, right):
        ordered.extend(sorted([b for b in remaining if b in column], key=lambda b: b.bbox[1]))
    return ordered, 2


def _table_regions(
    page: pymupdf.Page, blocks: list[tuple[object, ...]]
) -> list[tuple[float, float, float, float]]:
    """Find ruled tables, then try text alignment below each table caption.

    Restrict text-based detection to the caption-to-heading region so prose
    columns are not relabelled as table cells. Avoid scanning unrelated plots.
    """
    captions = [b for b in blocks if b[6] == 0 and re.match(r"^Table\s+\d+", str(b[4]), re.I)]
    if not captions:
        return []
    regions = [table.bbox for table in page.find_tables().tables]
    for caption in captions:
        bottom = float(caption[3])
        next_headings = [
            float(b[1])
            for b in blocks
            if b[6] == 0
            and float(b[1]) > bottom
            and NUMBERED.fullmatch(" ".join(str(b[4]).split()))
        ]
        end = min(next_headings, default=page.rect.height)
        if any(bottom <= (r[1] + r[3]) / 2 <= end for r in regions):
            continue
        clip = (page.rect.x0, bottom, page.rect.x1, end)
        regions.extend(table.bbox for table in page.find_tables(strategy="text", clip=clip).tables)
    return regions


def extract_pdf(path: Path, config: RetrievalSettings) -> Extraction:
    passages: list[Passage] = []
    diagnostics: list[PageDiagnostic] = []
    section_path: dict[int, str] = {}
    section: str | None = None
    zone: ChunkKind = "text"
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            styles = page.get_text("dict")["blocks"]
            sizes = [
                span["size"]
                for block in styles
                for line in block.get("lines", [])
                for span in line["spans"]
                if span["text"].strip()
            ]
            body_size = median(sizes) if sizes else 0
            raw_blocks = page.get_text("blocks")
            table_rects = _table_regions(page, raw_blocks)
            blocks: list[Passage] = []
            warnings: list[str] = []
            for raw in raw_blocks:
                if raw[6] != 0:
                    continue
                text = raw[4].strip()
                if not text or text.isdigit() or STAMP.search(text):
                    continue
                bbox = tuple(float(v) for v in raw[:4])
                # Preserve line breaks in tables, unwrap prose later.
                is_table = any(
                    rect[0] <= (bbox[0] + bbox[2]) / 2 <= rect[2]
                    and rect[1] <= (bbox[1] + bbox[3]) / 2 <= rect[3]
                    for rect in table_rects
                )
                kind: ChunkKind = "table" if is_table else "text"
                if CAPTION.match(text):
                    kind = "caption"
                blocks.append(
                    Passage(page=page_index + 1, section=None, kind=kind, text=text, bbox=bbox)
                )
            ordered, columns = _reading_order(blocks, page.rect.width, config)
            for block in ordered:
                flat = " ".join(block.text.split())
                heading = NUMBERED.fullmatch(flat)
                spans = [
                    span
                    for style in styles
                    if tuple(style["bbox"]) == block.bbox
                    for line in style.get("lines", [])
                    for span in line["spans"]
                ]
                prominent = any(
                    s["size"] >= body_size * config.heading_font_ratio or s["flags"] & 16
                    for s in spans
                )
                if flat.lower() in {"references", "bibliography"}:
                    section, zone = flat, "references"
                    section_path.clear()
                elif (
                    re.match(r"^appendix(?:\s|$)", flat, re.I)
                    and len(flat) < config.max_heading_chars
                ):
                    section, zone = flat, "appendix"
                    section_path.clear()
                elif heading and (prominent or len(flat) < config.max_heading_chars):
                    number = heading[1].rstrip(".")
                    level = number.count(".")
                    if number[0].isalpha():
                        zone = "appendix"
                    elif zone == "references":
                        zone = "text"
                    section_path = {k: v for k, v in section_path.items() if k < level}
                    section_path[level] = flat
                    section = " > ".join(section_path[k] for k in sorted(section_path))
                block.section = section
                if zone in {"references", "appendix"} and block.kind == "text":
                    block.kind = zone
                if block.kind != "table":
                    block.text = re.sub(r"(?<=\w)-\n(?=\w)", "", block.text)
                    block.text = " ".join(block.text.split())
                if "\ufffd" in block.text:
                    warnings.append("replacement characters: inspect extraction")
                if any(ord(char) < 32 and char not in "\n\r\t" for char in block.text):
                    warnings.append("control characters in mathematical extraction: inspect source")
                passages.append(block)
                if block.kind in {"table", "caption"}:
                    warnings.append(f"{block.kind}: inspect reading order and flattened cells")
            if not ordered:
                warnings.append("no text blocks: inspect PDF; OCR may be needed")
            if any(re.match(r"^Table\s+\d+", b.text, re.I) for b in ordered) and not table_rects:
                warnings.append("table caption without detected cells: inspect manually")
            diagnostics.append(
                PageDiagnostic(
                    page=page_index + 1,
                    blocks=len(ordered),
                    columns=columns,
                    tables_detected=len(table_rects),
                    warnings=sorted(set(warnings)),
                )
            )
    return Extraction(passages=passages, diagnostics=diagnostics)


def paragraph_windows(
    paragraphs: list[str],
    setting: ChunkSetting,
    encoding: tiktoken.Encoding,
    *,
    tolerance: float,
) -> list[str]:
    """Prefer paragraph boundaries within 15% of target; split oversized prose.

    All other windows are bounded token slices. Final partial windows are never
    padded. Overlap is the trailing configured tokens of the previous window.
    """
    text = "\n\n".join(paragraphs)
    tokens = token_ids(text, encoding)
    boundaries = [
        len(token_ids(text[: match.end()], encoding)) for match in re.finditer(r"\n\n", text)
    ]
    windows: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + setting.target_tokens, len(tokens))
        if end < len(tokens):
            candidates = [
                p for p in boundaries if start + setting.target_tokens * (1 - tolerance) <= p <= end
            ]
            if candidates:
                end = candidates[-1]
        # Do not cut UTF-8 codepoints: trim the token boundary until decodable.
        while end > start:
            try:
                window = encoding.decode_bytes(tokens[start:end]).decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            raise ValueError("unable to find a valid UTF-8 token boundary")
        if window.strip():
            windows.append(window)
        if end == len(tokens):
            break
        start = end - setting.overlap
        while start < end:
            try:
                encoding.decode_bytes(tokens[start:end]).decode("utf-8")
                break
            except UnicodeDecodeError:
                start += 1
    return windows


def build_chunks(
    extraction: Extraction,
    paper: PaperSpec,
    setting: ChunkSetting,
    config: RetrievalSettings,
) -> list[Chunk]:
    encoding = get_encoding(config.tokenizer)
    # A page/section/kind boundary is deliberately a partial-chunk boundary.
    groups: list[list[Passage]] = []
    for passage in extraction.passages:
        if groups and (groups[-1][-1].page, groups[-1][-1].section, groups[-1][-1].kind) == (
            passage.page,
            passage.section,
            passage.kind,
        ):
            groups[-1].append(passage)
        else:
            groups.append([passage])
    chunks: list[Chunk] = []
    copied: set[int] = set()
    for source_index, group in enumerate(groups):
        first = group[0]
        if first.kind not in {"table", "caption"}:
            continue
        if len(token_ids("\n\n".join(p.text for p in group), encoding)) > config.short_block_tokens:
            continue
        nearby = [
            i
            for i, target in enumerate(groups)
            if target[0].kind == "text"
            and target[0].page == first.page
            and target[0].section == first.section
        ]
        if nearby:
            target_index = min(nearby, key=lambda i: (abs(i - source_index), i))
            copies = [p.model_copy(update={"kind": "text"}) for p in group]
            if target_index < source_index:
                groups[target_index].extend(copies)
            else:
                groups[target_index][0:0] = copies
            copied.add(source_index)
    for group_index, group in enumerate(groups):
        first = group[0]
        paragraphs = [p.text for p in group]
        for text in paragraph_windows(
            paragraphs, setting, encoding, tolerance=config.chunk_tolerance
        ):
            chunks.append(
                Chunk(
                    **paper.model_dump(),
                    chunk_id=f"{paper.paper_id}-v{paper.arxiv_version}-p{first.page}-c{len(chunks)}",
                    section=first.section,
                    page_start=first.page,
                    page_end=group[-1].page,
                    kind=first.kind,
                    text=text,
                    n_tokens=len(token_ids(text, encoding)),
                )
            )
        # Short non-prose blocks are also available as prose evidence, as specified.
        if first.kind in {"table", "caption"} and group_index not in copied:
            text = "\n\n".join(paragraphs)
            count = len(token_ids(text, encoding))
            if count <= config.short_block_tokens:
                chunks.append(
                    Chunk(
                        **paper.model_dump(),
                        chunk_id=f"{paper.paper_id}-v{paper.arxiv_version}-p{first.page}-c{len(chunks)}",
                        section=first.section,
                        page_start=first.page,
                        page_end=group[-1].page,
                        kind="text",
                        text=text,
                        n_tokens=count,
                    )
                )
    return chunks


def extraction_summary(extraction: Extraction) -> dict[str, object]:
    return {
        "pages": [d.model_dump() for d in extraction.diagnostics],
        "kinds": dict(Counter(p.kind for p in extraction.passages)),
        "inspection_passages": [
            p.model_dump()
            for p in extraction.passages
            if re.search(r"3\.2\.3|Table 1\b|latency|dynamic chunking|methodology", p.text, re.I)
        ],
    }
