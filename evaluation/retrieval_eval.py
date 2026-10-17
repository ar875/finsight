"""Retrieval evaluation (no LLM calls): hit@k, recall@k, MRR, nDCG@k, candidate recall.

    python -m evaluation.retrieval_eval                 # the current config.yaml setup
    python -m evaluation.retrieval_eval --ablations     # Phase 1 -> Phase 2, one upgrade at a time
                                                        # (needs the Phase 1 and Phase 2 indexes built)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from statistics import mean

from finsight import RAGPipeline, deep_update, load_config
from finsight.config import BASELINE_OVERRIDES, PHASE2_OVERRIDES

from .common import RESULTS_DIR, load_golden, ref_matches


def retrieval_scores(refs: list[dict], candidates: list[dict], evidence: list[dict], k: int) -> dict:
    top = refs[:k]
    flags = [any(ref_matches(r, e) for e in evidence) for r in top]
    found = [any(ref_matches(r, e) for r in top) for e in evidence]
    first = next((i for i, f in enumerate(flags, 1) if f), None)
    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(evidence))))
    return {"hit@k": float(any(flags)), "recall@k": sum(found) / len(evidence),
            "mrr": 1 / first if first else 0.0, "ndcg@k": min(1.0, dcg / ideal) if ideal else 0.0,
            "candidate_recall": sum(any(ref_matches(r, e) for r in candidates) for e in evidence) / len(evidence)}


def evaluate_retrieval(pipeline, questions, k) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        if not q["evidence"]:
            continue                      # unanswerable questions have no evidence to find
        final, cands, _ = pipeline.search(q["question"])
        rows.append({"id": q["id"], "type": q["type"],
                     **retrieval_scores([c.ref() for c in final], [c.ref() for c in cands], q["evidence"], k)})
    summary = {m: round(mean(r[m] for r in rows), 3) for m in rows[0] if m not in ("id", "type")}
    return summary, rows


_NO = {"use_reranker": False, "company_filter": False, "soft_filters": False,
       "decompose_multi_company": False, "expand_query": False}
ABLATIONS = [   # each line adds ONE upgrade, so you can see what each one contributes
    ("1. Phase 1 baseline (fixed chunks, dense)", BASELINE_OVERRIDES),
    ("2. + structure-aware chunking", deep_update(PHASE2_OVERRIDES, {"retrieval": {"mode": "dense", **_NO}})),
    ("3. + hybrid search (BM25 + RRF)", deep_update(PHASE2_OVERRIDES, {"retrieval": _NO})),
    ("4. + company filter & decomposition", deep_update(PHASE2_OVERRIDES, {"retrieval": {
        "use_reranker": False, "soft_filters": False, "expand_query": False}})),
    ("5. + year/doc-type boost", deep_update(PHASE2_OVERRIDES, {"retrieval": {"use_reranker": False, "expand_query": False}})),
    ("6. + query expansion", deep_update(PHASE2_OVERRIDES, {"retrieval": {"use_reranker": False}})),
    ("7. + reranker (= Phase 2)", PHASE2_OVERRIDES),
    ("8. + unit headers & flat table headers (= Phase 4, current)", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--profile", help="named profile from config.yaml, e.g. light")
    args = ap.parse_args()
    questions = load_golden()
    setups = ABLATIONS if args.ablations else [(f"profile {args.profile}" if args.profile else "current config", {})]
    results = {}
    for name, overrides in setups:
        cfg = load_config(overrides=overrides, profile=args.profile)
        cfg["retrieval"]["final_k"] = max(args.k, cfg["retrieval"]["final_k"])     # same k for every setup (fair)
        t = time.time()
        summary, rows = evaluate_retrieval(RAGPipeline(cfg), questions, args.k)
        results[name] = {"summary": summary, "rows": rows}
        print(f"{name:<62} {summary}  ({time.time() - t:.0f}s)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / ("retrieval_ablation.json" if args.ablations else f"retrieval_{args.profile or 'current'}.json")
    out.write_text(json.dumps({n: r["summary"] for n, r in results.items()}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
