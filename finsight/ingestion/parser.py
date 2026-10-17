"""PDF/text parsing: headings, tables (as Markdown), header/footer removal, speaker turns."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path

import pymupdf

from ..schema import Element

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
SPEAKER_RE = re.compile(r"^[A-Z][A-Za-z.\-' ]{1,40}:\s")      # "Rajesh Gopinathan: ..." in transcripts
MAX_PARAGRAPH_WORDS = 160


def _cell(value) -> str:
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip().replace("|", "/")


def rows_to_markdown(rows: list[list]) -> str:
    clean = [[_cell(c) for c in row] for row in rows if row and any(_cell(c) for c in row)]
    if len(clean) < 2:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in clean[1:]])


def _normalise(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())       # page numbers become '#'


NOT_HEADINGS = re.compile(r"^(total|subject|particulars|notes?|sr\.? ?no\.?|s\.? ?no\.?|amount|year ended.*|"
                          r"as at.*|(march|mar) 31.*|in ₹.*|₹ in .*|\(.*\)|continued)$", re.I)


def _is_heading(text: str, size: float, bold: bool, body_size: float, allow_bold: bool = True) -> bool:
    """A heading is short, mostly letters, not a sentence, and either bigger than body text or bold Title/UPPER case."""
    words = text.split()
    if not words or len(words) > 12 or len(text) > 100 or text.endswith((".", ",", ";", ":")):
        return False
    chars = text.replace(" ", "")
    if sum(ch.isalpha() for ch in chars) < 0.7 * len(chars):          # mostly numbers / symbols: chart label
        return False
    if "%" in text or NOT_HEADINGS.match(text.strip()):
        return False
    if size >= body_size * 1.2:
        return True
    return allow_bold and bold and len(words) <= 8 and (text.isupper() or text.istitle())


class _Paragraphs:
    """Joins lines into paragraphs; never crosses a page, so citations stay accurate."""

    def __init__(self, elements: list, joiner: str = " "):
        self.elements, self.joiner, self.lines, self.page = elements, joiner, [], 1

    def add(self, text: str, page: int) -> None:
        if not self.lines:
            self.page = page
        self.lines.append(text)
        if len(self.joiner.join(self.lines).split()) >= MAX_PARAGRAPH_WORDS and text.rstrip().endswith((".", "?", "!")):
            self.flush()

    def flush(self) -> None:
        if self.lines:
            self.elements.append(Element(self.page, "text", self.joiner.join(self.lines).strip()))
            self.lines = []


def parse_pdf(path, detect_tables: bool = True, allow_bold_headings: bool = True) -> list[Element]:
    doc = pymupdf.open(path)
    pages, sizes = [], []      # pages: list of (y, kind, text, size, bold, in_margin)
    for page in doc:
        items, table_rects, height = [], [], page.rect.height or 1.0
        if detect_tables:
            try:
                for table in page.find_tables().tables:
                    md = rows_to_markdown(table.extract())
                    if md:
                        rect = pymupdf.Rect(table.bbox)
                        table_rects.append(rect)
                        items.append((rect.y0, "table", md, 0.0, False, False))
            except Exception:
                pass
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                rect = pymupdf.Rect(line["bbox"])
                centre = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
                if any(r.contains(centre) for r in table_rects):
                    continue                     # already inside a table
                size = max(s["size"] for s in spans)
                bold = any((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                in_margin = rect.y0 < 0.08 * height or rect.y1 > 0.92 * height
                items.append((rect.y0, "line", " ".join(s["text"].strip() for s in spans), size, bold, in_margin))
                sizes.append(round(size, 1))
        items.sort(key=lambda x: x[0])
        pages.append(items)
    doc.close()

    body_size = statistics.median(sizes) if sizes else 10.0
    repeated = set()
    if len(pages) >= 5:   # lines in the top/bottom margin that repeat on many pages = headers/footers
        counts = Counter(t for items in pages for t in {_normalise(x[2]) for x in items if x[1] == "line" and x[5]})
        repeated = {t for t, c in counts.items() if c >= max(3, int(0.3 * len(pages))) and len(t) < 100}

    elements = []
    para = _Paragraphs(elements)
    for page_no, items in enumerate(pages, start=1):
        for _, kind, text, size, bold, in_margin in items:
            if kind == "table":
                para.flush()
                elements.append(Element(page_no, "table", text))
            elif in_margin and (_normalise(text) in repeated or re.fullmatch(r"(page\s*)?\d{1,4}", text.lower())):
                continue
            elif _is_heading(text, size, bold, body_size, allow_bold_headings):
                para.flush()
                elements.append(Element(page_no, "heading", text))
            else:
                if SPEAKER_RE.match(text):
                    para.flush()
                para.add(text, page_no)
        para.flush()
    return elements


def parse_text(text: str) -> list[Element]:
    elements = []
    para = _Paragraphs(elements)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            para.flush()
        elif line.startswith("#"):
            para.flush()
            elements.append(Element(1, "heading", line.lstrip("#").strip()))
        else:
            if SPEAKER_RE.match(line):
                para.flush()
            para.add(line, 1)
    para.flush()
    return elements


def parse_document(path, detect_tables: bool = True) -> list[Element]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        # transcripts print speaker names in bold: those are NOT section headings
        return parse_pdf(path, detect_tables, allow_bold_headings="earnings_call" not in path.stem)
    if path.suffix.lower() in {".txt", ".md"}:
        return parse_text(path.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: {path.name}")
