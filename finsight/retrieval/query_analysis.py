"""Question understanding: company, year, quarter, document type, and keyword-search synonyms."""
from __future__ import annotations

import re

from ..companies import COMPANIES

GUIDANCE_WORDS = re.compile(r"\b(guidance|outlook|expect|expects|expected|target|forecast)\b", re.I)
CALL_WORDS = re.compile(r"\b(call|concall|transcript|management (said|say|commentary)|analysts?|ceo said|cfo said)\b", re.I)
REPORT_WORDS = re.compile(r"\b(annual report|balance sheet|notes to accounts|directors'? report)\b", re.I)


def _fy(year: str) -> str:
    year = int(year)
    return f"FY{year + 2000 if year < 100 else year}"


def detect_companies(question: str) -> list[str]:
    found = []
    for code, info in COMPANIES.items():
        names = {info["name"], *info["aliases"]}
        for name in names:
            flags = 0 if (name.isupper() and len(name) <= 4) else re.I   # short codes like "TCS" are case-sensitive
            if re.search(rf"(?<![\w&]){re.escape(name)}(?![\w&])", question, flags):
                found.append(code)
                break
    return found


def detect_years(question: str) -> list[str]:
    years = []
    for m in re.finditer(r"\bFY\s?'?(\d{4}|\d{2})\b", question, re.I):          # FY2025, FY 2025, FY25
        years.append(_fy(m.group(1)))
    for m in re.finditer(r"\b(20\d{2})\s?[-–/]\s?(\d{2,4})\b", question):          # 2024-25
        years.append(_fy(m.group(2)[-2:]))
    for m in re.finditer(r"\b(?:march|mar)\s*31,?\s*(20\d{2})\b", question, re.I):  # March 31, 2025
        years.append(_fy(m.group(1)))
    return list(dict.fromkeys(years))


FINANCE_SYNONYMS = {     # different reports use different names for the same thing
    "net profit": "profit after tax PAT profit for the year net income",
    "revenue": "revenue from operations total income",
    "operating margin": "EBIT margin operating profit",
    "gross npa": "gross non-performing assets GNPA",
    "net npa": "net non-performing assets NNPA",
    "net interest margin": "NIM",
    "attrition": "employee turnover",
    "headcount": "employees workforce",
    "guidance": "outlook expect",
}


def expand_query(question: str) -> str:
    """Adds synonyms for keyword (BM25) search only; the embedding model gets the original question."""
    extra = [syn for term, syn in FINANCE_SYNONYMS.items() if term in question.lower()]
    return question + (" " + " ".join(extra) if extra else "")


def analyse_question(question: str) -> dict:
    years = detect_years(question)
    if years and GUIDANCE_WORDS.search(question):
        # guidance FOR a year is given at the END of the previous year (FY2026 guidance -> FY2025 Q4 call)
        years = list(dict.fromkeys(_fy(int(y[2:]) - 1) for y in years))
    quarter = re.search(r"\bQ([1-4])\b", question, re.I)
    doc_type = None
    if CALL_WORDS.search(question) or GUIDANCE_WORDS.search(question) or quarter:
        doc_type = "earnings_call"
    elif REPORT_WORDS.search(question):
        doc_type = "annual_report"
    return {"companies": detect_companies(question), "years": years,
            "quarter": quarter.group(1) if quarter else None, "doc_type": doc_type}
