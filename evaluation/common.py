"""Shared helpers for the evaluation scripts: files, pacing, and evidence matching."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"           # golden.jsonl (the reviewed test set) + accept_also.json
RESULTS_DIR = EVAL_DIR / "results"     # answers_*.jsonl, retrieval results
EXCLUDE = {"g017"}                     # g017: scanned transcript misread "100%" -> a data problem, not a system one


def load_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_golden(path=DATA_DIR / "golden.jsonl", exclude=EXCLUDE) -> list[dict]:
    questions = [q for q in load_jsonl(path) if q["id"] not in exclude]
    if not questions:
        raise FileNotFoundError(f"No test set at {path}. Copy golden.jsonl from the finsight-eval dataset there.")
    return questions


def parse_json(text: str) -> dict:
    """Lenient JSON parsing: handles ```json fences and extra text around the object."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


class Pacer:
    """Keeps LLM calls at least `seconds` apart (Groq free tier: 8K tokens per minute per model)."""

    def __init__(self, seconds: float):
        self.seconds, self.last = seconds, 0.0

    def wait(self):
        delay = self.seconds - (time.time() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.last = time.time()


def ref_matches(ref: dict, ev: dict) -> bool:
    """A retrieved chunk counts as correct if it is from the same document and its pages overlap the evidence."""
    return ref["doc_id"] == ev["doc_id"] and ref["page_start"] <= ev["page_end"] and ev["page_start"] <= ref["page_end"]
