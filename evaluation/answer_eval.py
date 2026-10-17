"""Answer evaluation: runs every test question through the full pipeline and scores the answers.
Uses about 3,000 tokens per question. Safe to stop and re-run: it continues where it left off.

    python -m evaluation.answer_eval --out evaluation/results/answers_v1.jsonl
    python -m evaluation.report evaluation/results/answers_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from finsight import RAGPipeline, load_config
from finsight.generation.llm import DailyLimitError, llm_from_config

from .common import DATA_DIR, Pacer, append_jsonl, load_golden, load_jsonl
from .scoring import score_answer


def evaluate_answers(pipeline, questions, path, judge=None, pause=30.0, accept_also=None, log=print) -> int:
    done = {r["id"] for r in load_jsonl(path)}
    todo = [q for q in questions if q["id"] not in done]
    log(f"{len(done)} already done, {len(todo)} to go ...")
    pacer, judge_pacer = Pacer(pause), Pacer(10)
    for i, q in enumerate(todo, 1):
        pacer.wait()
        try:
            r = pipeline.ask(q["question"])
            s = score_answer(q, r, judge, judge_pacer, accept_also)
        except DailyLimitError as exc:
            log(f"\nSTOPPED: daily token limit reached.\n{str(exc)[:220]}\n{i - 1} new results saved. Re-run later.")
            return i - 1
        append_jsonl(path, {"id": q["id"], "type": q["type"], "question": q["question"], "reference": q["answer"],
                            "answer": r["answer"], "citations": [c["label"] for c in r["citations"]],
                            "top_rerank_score": r.get("top_rerank_score"), "latency_ms": r["latency_ms"],
                            "tokens": r.get("tokens", {}), **s})
        if i % 10 == 0:
            log(f"  {i}/{len(todo)}")
    log("Done.")
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="answers file (.jsonl); existing results are kept")
    ap.add_argument("--pause", type=float, default=30.0, help="seconds between questions (free-tier rate limit)")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--profile", help="named profile from config.yaml, e.g. light")
    args = ap.parse_args()
    cfg = load_config(profile=args.profile)
    print("Reranker:", cfg["retrieval"]["reranker_model"], "| gate:", cfg["generation"]["min_rerank_score"])
    accept = DATA_DIR / "accept_also.json"
    accept_also = {k: v for k, v in json.loads(accept.read_text()).items() if not k.startswith("_")} if accept.exists() else {}
    judge = llm_from_config({**cfg["generation"], "max_tokens": 800}, model=args.judge_model)
    evaluate_answers(RAGPipeline(cfg), load_golden(), args.out, judge, args.pause, accept_also)


if __name__ == "__main__":
    main()
