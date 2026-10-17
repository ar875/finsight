"""Document metadata from file names: TCS_FY2024Q4_earnings_call.pdf -> company, year, quarter, doc type."""
from __future__ import annotations

import re
from pathlib import Path

from ..companies import COMPANIES

FILENAME_RE = re.compile(
    r"^(?P<company>[A-Za-z0-9&-]+)_(?P<year>FY\d{4})(?:Q(?P<quarter>[1-4]))?_(?P<doc_type>annual_report|earnings_call)$"
)


def extract_metadata(path) -> dict:
    """TCS_FY2024Q4_earnings_call.pdf -> company, year, quarter, doc_type, sector, title"""
    path = Path(path)
    match = FILENAME_RE.match(path.stem)
    if not match:
        return {"doc_id": path.stem, "title": path.stem, "doc_type": "unknown", "matched_pattern": False}
    meta = {k: v for k, v in match.groupdict().items() if v}
    info = COMPANIES.get(meta["company"], {})
    quarter = f" Q{meta['quarter']}" if meta.get("quarter") else ""
    meta.update(doc_id=path.stem, sector=info.get("sector", ""), matched_pattern=True,
                title=f"{info.get('name', meta['company'])} {meta['year']}{quarter} "
                      f"{meta['doc_type'].replace('_', ' ').title()}")
    return meta
