"""Page units: reads '(₹ in '000s)'-style lines straight from the PDF page text."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

from ..utils import file_hash

_SYM = r"(?:₹|`|Rs\.?|INR|\bC\b|\bI\b|\bH\b|\bJ\b)"            # PDFs often turn ₹ into ` C I H or J
_SCALE_WORDS = r"(?:[\'‘’]\s?000\'?s?\b|thousands?\b|lakhs?\b|crores?\b|millions?\b|billions?\b)"
_UNIT_PATTERNS = [
    re.compile(r"\b(?:US\$|USD)\s*(?:in\s+)?(millions?|billions?)\b", re.I),                          # US$ million
    re.compile(rf"{_SYM}\s*(?:in\s+)?({_SCALE_WORDS})", re.I),                                       # ₹ in '000s, ` crore
    re.compile(rf"\(\s*(?:all\s+)?(?:amounts?\s+)?in\s+(?:\S{{1,3}}\s*)?({_SCALE_WORDS})", re.I),    # (All amounts in ₹ crores
]


def _normalise_unit(word: str, usd: bool) -> str:
    w = word.lower().strip("'‘’ s")
    scale = "thousand" if "000" in w or w.startswith("thousand") else w.rstrip("s")
    return f"{'US$' if usd else '₹'} {scale}"


def detect_page_units(text: str) -> str | None:
    """'(₹ in '000s)' -> '₹ thousand'. Most common unit on the page wins; None if no unit line."""
    found = []
    for i, pattern in enumerate(_UNIT_PATTERNS):
        found += [_normalise_unit(m, usd=(i == 0)) for m in pattern.findall(text)]
    return Counter(found).most_common(1)[0][0] if found else None


def page_units_for(cfg, path) -> dict[int, str]:
    if Path(path).suffix.lower() != ".pdf":
        return {}
    cache = Path(cfg["data"]["parsed_dir"]) / f"{Path(path).stem}.{file_hash(path)[:12]}.units.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    units = {}
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            unit = detect_page_units(page.get_text("text"))
            if unit:
                units[page_no] = unit
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(units))
    return units
