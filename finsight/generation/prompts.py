"""Answer prompt, source formatting and citation handling."""
from __future__ import annotations

import re

ANSWER_SYSTEM = """You are a careful financial analyst answering questions about company annual reports and earnings call transcripts.
Answer using ONLY the numbered sources provided.

Rules:
1. End every sentence that states a fact with its citation in square brackets, like [1] or [2][3].
2. Always name the company and the period in your answer (e.g. "HDFC Bank's gross NPA ratio as of March 31, 2025 was ...").
3. Only use a source if it is about the company and period asked. Annual reports also show previous-year figures; use them only if clearly labelled with the right year.
4. If the sources contain only part of the answer (for example, one company in a comparison), answer that part
   and clearly say what is missing. Reply with exactly one line "NOT_FOUND: <short reason>" only when the
   sources contain none of the answer.
5. Never use outside knowledge and never guess numbers. You may do simple arithmetic on cited numbers
   (differences, growth %); if you do, say "calculated".
6. Copy numbers exactly and ALWAYS write their unit and scale. A source header like "[... | amounts in ₹ thousand]"
   or a table title like "(₹ in '000s)" gives the unit of that source's MONEY amounts: write "₹1,202,409,132 thousand",
   never just "₹1,202,409,132". Percentages, ratios and counts keep their own units (13.4% stays 13.4%).
   Never assume or convert currency (₹ is not US$). If no unit is stated anywhere, say "unit not stated".
   Read row and column headers carefully.
7. For comparisons, give each company's figure with its own citation and unit. If the units differ, say so.
Keep the answer short (at most 6 sentences)."""


def format_sources(chunks) -> str:
    return "\n\n".join(f"[{i}] {c.chunk.source_label()}\n{c.chunk.text}" for i, c in enumerate(chunks, start=1))


def build_messages(question, chunks) -> list[dict]:
    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"}]


def normalize_citations(answer: str) -> str:
    answer = answer.replace("【", "[").replace("】", "]")
    return re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", answer)


def extract_citations(answer: str) -> list[int]:
    numbers = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", normalize_citations(answer)):
        for n in group.split(","):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def is_refusal(answer: str) -> bool:
    """NOT_FOUND anywhere counts: the model sometimes writes a sentence first, then NOT_FOUND (Phase 4, g100)."""
    return "NOT_FOUND" in answer.upper()
