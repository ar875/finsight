"""Answer scoring (scorer v2).

Numeric answers are checked exactly: every number in the reference must appear in the answer after unit
conversion (₹16,103,480,210 thousand = ₹16,103.48 billion), with 0.5% tolerance for rounding, and a ₹ vs US$
mix-up is a currency error. Text answers are graded by an LLM judge. Unanswerable questions must be refused.
"""
from __future__ import annotations

import re

from .common import parse_json, ref_matches

_DATE_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|"
                      r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", re.I)
_YEAR_RE = re.compile(r"\b(?:FY\s?'?)?(?:19|20)\d{2}(?:\s?[-–/]\s?\d{2,4})?\b|\bFY\s?'?\d{2}\b|\bQ[1-4]\b", re.I)

NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
                "forty": 40, "fifty": 50, "hundred": 100}
SCALE = {"thousand": 1e3, "lakh": 1e5, "million": 1e6, "crore": 1e7, "billion": 1e9}
_AMOUNT_RE = re.compile(r"(₹|Rs\.?|INR|US\$|USD|\$)?\s?(?<![\w.])(\d+(?:\.\d+)?)\s*"
                        r"(thousand|lakhs?|million|mn\b|crores?|cr\b|billion|bn\b)?", re.I)

JUDGE_PROMPT = """Grade the candidate answer against the reference answer for this question.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
Score 1 if the candidate states the key facts of the reference (wording may differ), 0.5 if partly correct,
0 if wrong, missing or a refusal. Return JSON only: {"score": 0 | 0.5 | 1, "reason": "<one sentence>"}"""


def _clean(text: str) -> str:
    t = _YEAR_RE.sub(" ", _DATE_RE.sub(" ", text or ""))
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    for word, value in NUMBER_WORDS.items():
        t = re.sub(rf"\b{word}\b", str(value), t, flags=re.I)
    return t


def amounts(text: str) -> list[tuple[float, str]]:
    """Every number as (absolute value, currency): 'US$16,103.48 billion' -> (1.610348e13, 'USD')."""
    out = []
    for cur, num, scale in _AMOUNT_RE.findall(_clean(text)):
        word = (scale or "").lower().rstrip("s")
        word = {"mn": "million", "cr": "crore", "bn": "billion"}.get(word, word)
        cur = "USD" if cur.upper() in ("US$", "USD", "$") else "INR" if cur else ""
        out.append((float(num) * SCALE.get(word, 1), cur))
    return out


def score_numeric(reference: str, answer: str) -> dict:
    """Each reference number must appear in the answer (same value after unit conversion, 0.5% tolerance).
    If both sides state a currency for that number and they differ (₹ vs US$), it is a currency error."""
    ref, ans = amounts(reference), amounts(answer)
    if not ref:
        return {}
    hits, currency_error = 0, False
    for value, cur in ref:
        match = [c for v, c in ans if abs(v - value) <= 0.005 * max(abs(value), 1e-9)]
        if match:
            hits += 1
            if cur and all(c and c != cur for c in match):
                currency_error = True
    share = hits / len(ref)
    correct = 0.0 if currency_error else (1.0 if share == 1 else 0.5 if share > 0 else 0.0)
    return {"correct": correct, "currency_error": currency_error, "number_match": round(share, 2)}


def score_answer(q: dict, r: dict, judge=None, pacer=None, accept_also: dict | None = None) -> dict:
    """q: test-set question, r: pipeline.ask() result. accept_also: {id: verified alternative answer}."""
    s = {"refused": r["refused"], "grounded": r["grounded"], "gated": r.get("gated", False),
         "n_citations": len(r["citations"])}
    if q["type"] == "unanswerable":
        s.update(correct=1.0 if r["refused"] else 0.0, method="refusal")
        return s
    if r["refused"] or r.get("error"):
        s.update(correct=0.0, method="refused" if r["refused"] else "error")
    else:
        numeric = score_numeric(q["answer"], r["answer"])
        if numeric:
            alt = score_numeric(accept_also[q["id"]], r["answer"]) if q["id"] in (accept_also or {}) else {}
            if alt and alt["correct"] > numeric["correct"]:     # verified alternative (consolidated vs standalone)
                numeric = {**alt, "accepted_alternative": True}
            s.update(numeric, method="numbers")
        elif judge is not None:
            if pacer:
                pacer.wait()
            prompt = (JUDGE_PROMPT.replace("{question}", q["question"]).replace("{reference}", q["answer"])
                      .replace("{answer}", r["answer"]))
            verdict = parse_json(judge.chat([{"role": "user", "content": prompt}]).text)
            s.update(correct=float(verdict.get("score", 0) or 0), judge_reason=verdict.get("reason", ""), method="judge")
        else:
            s.update(correct=None, method="not scored")
    cited = r["citations"]
    if cited and not r["refused"]:
        s["citation_precision"] = sum(any(ref_matches(c, e) for e in q["evidence"]) for c in cited) / len(cited)
    s["evidence_in_final"] = any(ref_matches(x, e) for x in r["sources"] for e in q["evidence"])
    s["evidence_in_candidates"] = any(ref_matches(x, e) for x in r["candidates"] for e in q["evidence"])
    return s
