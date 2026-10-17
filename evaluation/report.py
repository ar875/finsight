"""Summary table + failure analysis for one or more answers files (no LLM calls).

    python -m evaluation.report evaluation/results/answers_phase2_v2.jsonl evaluation/results/answers_phase4_final.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_jsonl


def answer_summary(path) -> dict:
    df = pd.DataFrame(load_jsonl(path))
    ans, una = df[df.type != "unanswerable"], df[df.type == "unanswerable"]
    answered = ans[~ans.refused]
    lat = [r["total"] for r in df.latency_ms if isinstance(r, dict) and "total" in r]
    return {"questions": len(df),
            "answer accuracy": round(ans.correct.mean(), 3),
            "table accuracy": round(ans[ans.type == "table"].correct.mean(), 3),
            "currency errors": int(ans.get("currency_error", pd.Series(dtype=object)).eq(True).sum()),
            "false refusal rate": round(ans.refused.mean(), 3),
            "correct refusal rate": round(una.refused.mean(), 3) if len(una) else None,
            "citation precision": round(answered.citation_precision.dropna().mean(), 3)
            if "citation_precision" in answered else None,
            "median latency (s)": round(float(np.median(lat)) / 1000, 2) if lat else None,
            "avg tokens": int(np.mean([t.get("input", 0) + t.get("output", 0) for t in df.tokens]))}


def failure_category(row) -> str:
    if row["type"] == "unanswerable":
        return "ok" if row["refused"] else "hallucinated (answered an unanswerable question)"
    if row["correct"] == 1:
        return "ok"
    if row.get("currency_error") is True:
        return "currency error (right number, wrong currency)"
    if not row.get("evidence_in_candidates", True):
        return "search miss (evidence never retrieved)"
    if not row.get("evidence_in_final", True):
        return "reranker drop (retrieved, then ranked out)"
    if row["refused"]:
        return "false refusal (evidence was given to the LLM)"
    if row["correct"] == 0.5:
        return "partly correct (some facts missing)"
    return "LLM error (evidence given, answer wrong)"


def md(df) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return df.to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    print(md(pd.DataFrame({Path(p).stem: answer_summary(p) for p in args.paths})))
    last = pd.DataFrame(load_jsonl(args.paths[-1]))
    last["failure"] = last.apply(failure_category, axis=1)
    ans = last[last.type != "unanswerable"]
    print("\nAccuracy by question type:")
    print(md(ans.groupby("type").correct.agg(["mean", "count"]).round(3)))
    print("\nRemaining failures:")
    for _, row in last[last.failure != "ok"].iterrows():
        print(f"  [{row['failure'].split(' (')[0]}] {row['id']} ({row['type']}): {row['question'][:90]}")


if __name__ == "__main__":
    main()
