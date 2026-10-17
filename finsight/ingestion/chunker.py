"""Chunking strategies: fixed (Phase 1), recursive, structure-aware (sections, tables, context headers)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..schema import Chunk, Element

def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))       # ~1.3 tokens per English word


@dataclass
class _Unit:
    text: str
    page: int
    section: str
    kind: str


def _to_units(elements: list[Element]) -> list[_Unit]:
    section, units = "", []
    for el in elements:
        if el.kind == "heading":
            section = el.text[:150]
        units.append(_Unit(el.text, el.page, section, el.kind))
    return units


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(₹$])", text) if s.strip()]


def _split_long(unit: _Unit, max_tokens: int) -> list[_Unit]:
    if approx_tokens(unit.text) <= max_tokens:
        return [unit]
    if unit.kind == "table":        # split by rows, repeating the header row in every piece
        lines = unit.text.split("\n")
        header, pieces, cur = lines[:2], [], []
        for row in lines[2:]:
            if cur and approx_tokens("\n".join(header + cur + [row])) > max_tokens:
                pieces.append(cur)
                cur = []
            cur.append(row)
        if cur:
            pieces.append(cur)
        return [_Unit("\n".join(header + p), unit.page, unit.section, "table") for p in pieces]
    parts, step = [], max(1, int(max_tokens / 1.3))
    for sentence in _split_sentences(unit.text):
        if approx_tokens(sentence) <= max_tokens:
            parts.append(sentence)
        else:
            words = sentence.split()
            parts += [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
    return [_Unit(p, unit.page, unit.section, unit.kind) for p in parts]


def _pack(units: list[_Unit], max_tokens: int, overlap_tokens: int) -> list[list[_Unit]]:
    """Greedily pack units up to max_tokens, carrying the last few units over as overlap."""
    expanded = [piece for u in units for piece in _split_long(u, max_tokens)]
    groups, cur, cur_tokens = [], [], 0
    for unit in expanded:
        t = approx_tokens(unit.text)
        if cur and cur_tokens + t > max_tokens:
            groups.append(cur)
            keep, kept = [], 0
            for prev in reversed(cur):
                pt = approx_tokens(prev.text)
                if kept + pt > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += pt
            cur, cur_tokens = keep, kept
        cur.append(unit)
        cur_tokens += t
    if cur:
        groups.append(cur)
    return groups


def _content_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()


def _make_chunk(group, doc_id, idx, metadata, header="", content_type=None) -> Chunk:
    body = "\n".join(u.text for u in group)
    pages = [u.page for u in group]
    ctype = content_type or ("table" if all(u.kind == "table" for u in group) else "text")
    return Chunk(chunk_id=f"{doc_id}::{idx:05d}", doc_id=doc_id, text=f"{header}\n{body}" if header else body,
                 page_start=min(pages), page_end=max(pages),
                 section=next((u.section for u in group if u.section), ""), content_type=ctype,
                 metadata={**metadata, "doc_id": doc_id, "content_hash": _content_hash(body)})


def _is_real_table(md: str) -> bool:
    """Rejects 1-column boxes, 1-row boxes, and bar-chart labels mis-detected as tables."""
    lines = md.split("\n")
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines if not line.startswith("| ---")]
    if len(rows[0]) < 2 or (len(rows) < 3 and approx_tokens(md) < 25):
        return False
    cells = [c for r in rows for c in r]
    if sum(not c for c in cells) / len(cells) > 0.4:                 # mostly empty cells
        return False
    header = [c for c in rows[0] if c]
    is_year = lambda c: re.fullmatch(r"(FY\s?)?(19|20)\d{2}(-\d{2})?", c, re.I)
    if header and all(re.fullmatch(r"[\d,.\s%()₹`-]+", c) and not is_year(c) for c in header):
        return False                                                 # header is only numbers: chart labels
    return True


_NUMBER_CELL = re.compile(r"^[\s\d,.\-–()%₹`]*$")
_YEAR_CELL = re.compile(r"^(FY\s?)?(19|20)\d{2}(\s?[-–]\s?\d{2,4})?\*?$", re.I)


def flatten_table_header(md: str) -> str:
    """Merges a 2-row header into one, so every column says what it is.
    | | FY 2026 | | |          ->   | | FY 2026 Male % | FY 2026 Female % | FY 2026 Total % |
    | | Male % | Female % | Total % |"""
    lines = md.split("\n")
    if len(lines) < 4:
        return md
    cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
    top, second = cells(lines[0]), cells(lines[2])
    if len(top) != len(second) or not any(top) or "" not in top[1:]:
        return md                                   # top row does not span columns: nothing to merge
    filled = [c for c in second if c]
    if len(filled) < 0.5 * len(second):
        return md                                   # second row is mostly empty: it is data, not a header
    numbers = [c for c in filled if _NUMBER_CELL.match(c) and not _YEAR_CELL.match(c)]
    if len(numbers) > 0.2 * len(filled):
        return md                                   # second row holds numbers: it is data, not a header
    spanned, last = [], ""
    for i, c in enumerate(top):
        last = c or (last if i > 0 else "")
        spanned.append(last)
    header = [" ".join(x for x in (a, b) if x) for a, b in zip(spanned, second)]
    return "\n".join(["| " + " | ".join(header) + " |", lines[1]] + lines[3:])


def chunk_elements(elements, doc_id, metadata, strategy="structure", chunk_size=512, chunk_overlap=64,
                   min_section_tokens=200, page_units=None, flatten_headers=False) -> list[Chunk]:
    """page_units: {page: "₹ thousand"} adds the unit to chunk headers. flatten_headers merges 2-row table headers."""
    page_units = page_units or {}
    if flatten_headers:
        elements = [Element(e.page, e.kind, flatten_table_header(e.text)) if e.kind == "table" else e for e in elements]
    if strategy == "fixed":           # Phase 1: word windows, ignores structure
        words = [(w, el.page) for el in elements for w in el.text.split()]
        size_w, overlap_w = max(1, int(chunk_size / 1.3)), int(chunk_overlap / 1.3)
        chunks = []
        for i in range(0, len(words), max(1, size_w - overlap_w)):
            window = words[i:i + size_w]
            chunk = _make_chunk([_Unit(" ".join(w for w, _ in window), window[0][1], "", "text")],
                                doc_id, len(chunks), metadata)
            chunk.page_end = window[-1][1]
            chunks.append(chunk)
            if i + size_w >= len(words):
                break
        return chunks

    units = _to_units(elements)
    if strategy == "recursive":       # paragraphs/sentences packed to size, no sections
        groups = _pack([u for u in units if u.kind != "heading"], chunk_size, chunk_overlap)
        return [_make_chunk(g, doc_id, i, metadata) for i, g in enumerate(groups)]
    if strategy != "structure":
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    # structure: follow sections, tables are separate chunks, every chunk gets a context header
    title, chunks, sections = metadata.get("title", doc_id), [], []
    for u in units:
        if u.kind == "table" and not _is_real_table(u.text):
            u.kind = "text"                     # layout boxes and chart labels are not real tables
        if not sections or u.section != sections[-1][0].section:
            sections.append([])
        sections[-1].append(u)
    merged = []                                 # merge tiny sections into the next one
    for sec in sections:
        if merged and sum(approx_tokens(u.text) for u in merged[-1] if u.kind != "table") < min_section_tokens:
            merged[-1].extend(sec)
        else:
            merged.append(sec)
    sections = merged
    def header_for(section, page=None):
        parts = [title] + ([section] if section else []) + ([f"amounts in {page_units[page]}"] if page in page_units else [])
        return "[" + " | ".join(parts) + "]"

    room = chunk_size - approx_tokens(header_for("x" * 60)) - (4 if page_units else 0)   # space for the header
    for sec_units in sections:
        text_buf = []

        def flush_text():
            if any(u.kind != "heading" for u in text_buf):
                for g in _pack(text_buf, room, chunk_overlap):
                    if any(u.kind != "heading" for u in g):
                        section = next((u.section for u in g if u.section), "")
                        chunks.append(_make_chunk(g, doc_id, len(chunks), metadata, header_for(section, g[0].page)))
            text_buf.clear()

        for u in sec_units:
            if u.kind == "table":            # a table is labelled with the section it actually sits in
                flush_text()
                for piece in _split_long(u, room):
                    chunks.append(_make_chunk([piece], doc_id, len(chunks), metadata,
                                              header_for(u.section, u.page), "table"))
            else:
                text_buf.append(u)
        flush_text()
    return chunks
