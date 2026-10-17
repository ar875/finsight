"""Shared by the FastAPI app (app/main.py) and the Streamlit app (streamlit_app.py)."""
from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from .config import ROOT

EXAMPLE_ANSWERS = ROOT / "data" / "example_answers.json"     # made by scripts/precompute_examples.py

EXAMPLES = [
    "What was TCS's revenue from operations in FY2025?",
    "What was HDFC Bank's gross NPA ratio as of March 31, 2025?",
    "What revenue growth guidance did Infosys give for FY2026?",
    "Compare the net profit of TCS and Wipro in FY2025",
    "What was Hindustan Unilever's total cash and cash equivalents as at 31 March 2025?",
    "How did ICICI Bank's total provisions change from Q4 FY2025 to Q4 FY2026?",
    "What was Wipro's revenue in FY2010?",
]
EVALUATION = {   # full profile (bge-reranker-base, top 30, GPU), 55-question test set, see README
    "questions": 55, "answer_accuracy": 0.889, "table_accuracy": 0.846, "retrieval_hit_at_6": 0.80,
    "correct_refusals": 1.0, "currency_errors": 0, "median_seconds": 1.6,
}
EVALUATION_LIGHT = {   # light profile = the public website (gte-reranker int8 ONNX, top 15, CPU), same test set
    "questions": 55, "answer_accuracy": 0.789, "table_accuracy": 0.692, "retrieval_hit_at_6": 0.844,
    "correct_refusals": 1.0, "currency_errors": 0, "median_seconds": 12.2,
}


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int):
        self.per_minute, self.per_day = per_minute, per_day
        self.visitors: dict[str, deque] = defaultdict(deque)
        self.day, self.today = time.strftime("%Y-%m-%d"), 0
        self.lock = threading.Lock()

    def check(self, visitor: str) -> str | None:
        """Returns an error message, or None if the question may use the LLM."""
        now = time.time()
        with self.lock:
            if time.strftime("%Y-%m-%d") != self.day:
                self.day, self.today = time.strftime("%Y-%m-%d"), 0
            q = self.visitors[visitor]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return f"You can ask {self.per_minute} questions a minute. Wait a moment, then ask again."
            if self.today >= self.per_day:
                return "Today's answer budget for this demo is used up. Passages are still shown below."
            q.append(now)
            self.today += 1
        return None


def document_links(path) -> dict[str, str]:
    links = {}
    if Path(path).exists():
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                q = f"Q{row['quarter']}" if row["quarter"] else ""
                links[f"{row['company']}_{row['year']}{q}_{row['doc_type']}"] = row["url"]
    return links
