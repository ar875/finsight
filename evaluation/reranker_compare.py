"""Compare rerankers on the test set without LLM calls: retrieval quality and the relevance-gate threshold.

    python -m evaluation.reranker_compare --models BAAI/bge-reranker-base cross-encoder/ms-marco-MiniLM-L-6-v2

For each reranker:
  * hit@k, recall@k, MRR, nDCG@k, candidate recall (same metrics as retrieval_eval)
  * the best reranker score of every question. A good gate threshold lies BELOW the scores of answerable questions
    whose evidence was found, and ABOVE the scores of unanswerable questions.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from statistics import mean

from finsight import RAGPipeline, load_config
from finsight.retrieval.reranker import get_reranker

from .common import RESULTS_DIR, load_golden
from .retrieval_eval import retrieval_scores

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def compare(model: str, questions: list[dict], k: int, top_n: int | None = None, max_length: int | None = None) -> dict:
    cfg = load_config(overrides={"retrieval": {"reranker_model": model, "final_k": k, "rerank_top_n": top_n,
                                               "rerank_max_length": max_length}})
    pipe = RAGPipeline(cfg)
    rows, t = [], time.time()
    for q in questions:
        final, cands, _ = pipe.search(q["question"])
        row = {"id": q["id"], "type": q["type"], "top": final[0].rerank_score if final else 0.0}
        if q["evidence"]:
            row.update(retrieval_scores([c.ref() for c in final], [c.ref() for c in cands], q["evidence"], k))
        rows.append(row)
    seconds = (time.time() - t) / len(questions)
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    found = [r for r in answerable if r["hit@k"] == 1]              # evidence reached the LLM: must not be gated
    unans = [r for r in rows if r["type"] == "unanswerable"]
    summary = {m: round(mean(r[m] for r in answerable), 3) for m in ("hit@k", "recall@k", "mrr", "ndcg@k", "candidate_recall")}
    gate = {t_: {"answerable_found_gated": sum(r["top"] < t_ for r in found),
                 "unanswerable_gated": sum(r["top"] < t_ for r in unans)} for t_ in THRESHOLDS}
    result = {"model": model, "rerank_top_n": top_n, "rerank_max_length": max_length, "summary": summary, "seconds_per_question": round(seconds, 2),
              "lowest_score_answerable_found": round(min(r["top"] for r in found), 4) if found else None,
              "unanswerable_scores": sorted(round(r["top"], 4) for r in unans), "gate": gate, "rows": rows}
    del pipe
    get_reranker.cache_clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rerank-top-n", type=int, default=None, help="re-rank only the best n candidates (default: all 30)")
    ap.add_argument("--rerank-max-length", type=int, default=None, help="tokens per (question, passage) pair")
    ap.add_argument("--grid", help="several 'top_n:max_length' settings for each model, e.g. 30:512,15:256,10:256")
    ap.add_argument("--out", default="reranker_compare.json")
    args = ap.parse_args()
    questions = load_golden()
    settings = ([tuple(int(x) for x in g.split(":")) for g in args.grid.split(",")] if args.grid
                else [(args.rerank_top_n, args.rerank_max_length)])
    results = []
    for model, (top_n, max_length) in [(m, st) for m in args.models for st in settings]:
        r = compare(model, questions, args.k, top_n, max_length)
        results.append(r)
        s = r["summary"]
        print(f"\n=== {model} | re-rank top {top_n or 'all'} at {max_length or 'full'} tokens  "
              f"({r['seconds_per_question']} s/question on this machine)")
        print(f"  hit@{args.k} {s['hit@k']} | recall {s['recall@k']} | MRR {s['mrr']} | nDCG {s['ndcg@k']} | "
              f"candidate recall {s['candidate_recall']}")
        print(f"  lowest top score of an answerable question whose evidence was found: {r['lowest_score_answerable_found']}")
        print(f"  top scores of the unanswerable questions: {r['unanswerable_scores']}")
        print("  gate threshold -> answerable-with-evidence wrongly gated | unanswerable gated")
        for t_, g in r["gate"].items():
            print(f"     {t_:<5} -> {g['answerable_found_gated']:>2} | {g['unanswerable_gated']}/{len(r['unanswerable_scores'])}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / args.out).write_text(json.dumps(results, indent=2))
    print("\nSaved", RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
