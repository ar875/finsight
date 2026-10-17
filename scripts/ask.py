"""Ask one question from the command line (needs LLM_API_KEY).

    python scripts/ask.py "What was TCS's revenue from operations in FY2025?"
    python scripts/ask.py --search-only "HDFC Bank gross NPA March 31, 2025"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsight import RAGPipeline, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--search-only", action="store_true", help="show retrieved passages, no LLM call")
    args = ap.parse_args()
    pipe = RAGPipeline(load_config())
    if args.search_only:
        final, _, info = pipe.search(args.question)
        print("Filters:", info["hard_filter"], "| boost:", info["soft_filter"])
        for i, c in enumerate(final, 1):
            print(f"[{i}] {c.chunk.source_label()}  (rerank {c.rerank_score})")
        return
    r = pipe.ask(args.question)
    print(r["answer"], "\n")
    for c in r["citations"]:
        print(f"  [{c['number']}] {c['label']}")
    print(f"\nLatency: {r['latency_ms']} | Tokens: {r.get('tokens')} | Top rerank score: {r.get('top_rerank_score')}")


if __name__ == "__main__":
    main()
