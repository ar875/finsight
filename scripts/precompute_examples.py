"""Answer the website's example questions once (light profile) and save them, so example clicks are instant
and use no LLM tokens on the website. The saved file also sets which examples the website shows, and in what order.

    python scripts/precompute_examples.py                          # answer the examples not saved yet
    python scripts/precompute_examples.py --force                  # answer all of them again
    python scripts/precompute_examples.py --swap "old" "new 1" "new 2"   # replace one example with the first
                                                                          # new question that gets a real answer
    python scripts/precompute_examples.py --remove "question"

Needs LLM_API_KEY. Re-run with --force after changing the index, the prompt or the light profile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsight import RAGPipeline, load_config  # noqa: E402
from finsight.web import EXAMPLE_ANSWERS, EXAMPLES  # noqa: E402

KEEP = ("question", "answer", "refused", "gated", "error", "citations", "sources", "filters",
        "top_rerank_score", "latency_ms", "tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--swap", nargs="+", metavar="Q", help="old question, then one or more candidates")
    ap.add_argument("--remove")
    args = ap.parse_args()

    saved = json.loads(EXAMPLE_ANSWERS.read_text(encoding="utf-8")) if EXAMPLE_ANSWERS.exists() else {}
    answers = {} if args.force else saved.get("answers", {})
    order = saved.get("order") or list(EXAMPLES)
    cfg = load_config(profile="light")
    pipe = RAGPipeline(cfg)
    if args.swap:
        old, candidates = args.swap[0], args.swap[1:]
        for j, new in enumerate(candidates):
            r = pipe.ask(new)
            print(f"Trying: {new}\n    {str(r['answer'])[:300]}\n    {r['latency_ms']}")
            if not (r["refused"] or r.get("gated") or r.get("error")):
                order = [new if q == old else q for q in order] if old in order else order + [new]
                answers.pop(old, None)
                answers[new] = {**{k: r.get(k) for k in KEEP}, "candidates_count": len(r["candidates"])}
                print("    -> using this one")
                break
            if j < len(candidates) - 1:
                time.sleep(10)
        else:
            print("No candidate was answered; examples unchanged")
    if args.remove:
        order = [q for q in order if q != args.remove]
        answers.pop(args.remove, None)

    todo = [q for q in order if q not in answers]
    print(f"{len(order)} examples, {len(todo)} to answer")
    for i, q in enumerate(todo, 1):
        r = pipe.ask(q)
        answers[q] = {**{k: r.get(k) for k in KEEP}, "candidates_count": len(r["candidates"])}
        print(f"[{i}/{len(todo)}] {q}\n    {str(r['answer'])[:200]}\n    {r['latency_ms']}")
        if i < len(todo):
            time.sleep(10)                                   # free-tier rate limit
    EXAMPLE_ANSWERS.write_text(json.dumps({"made_on": date.today().isoformat(), "profile": cfg.get("profile"),
                                           "reranker": cfg["retrieval"]["reranker_model"], "order": order,
                                           "answers": {q: answers[q] for q in order}},
                                          ensure_ascii=False, indent=1))
    print("Saved", EXAMPLE_ANSWERS, "| examples shown on the website:")
    for q in order:
        print("   ", "REFUSED " if answers[q].get("refused") else "answered", "|", q)


if __name__ == "__main__":
    main()
