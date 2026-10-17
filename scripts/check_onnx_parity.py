"""Do the ONNX models give the same numbers as the PyTorch ones? (run where both are installed, e.g. Kaggle)

    python scripts/check_onnx_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import load_golden  # noqa: E402
from finsight import RAGPipeline, load_config  # noqa: E402
from finsight.indexing.embedder import get_embedder  # noqa: E402
from finsight.retrieval.reranker import get_reranker  # noqa: E402


def main():
    full, light = load_config(), load_config(profile="light")
    questions = [q["question"] for q in load_golden()]

    torch_emb = get_embedder(full["embedding"]["model"], 64)
    onnx_emb = get_embedder(full["embedding"]["model"], 64, light["embedding"]["onnx_model"])
    a = np.vstack([torch_emb.embed_query(q) for q in questions])
    b = np.vstack([onnx_emb.embed_query(q) for q in questions])
    print(f"Embedder: {len(questions)} questions | lowest cosine(torch, onnx) = {float((a * b).sum(1).min()):.6f}")

    pipe = RAGPipeline(full)
    torch_rr = get_reranker(full["retrieval"]["reranker_model"], light["retrieval"].get("rerank_max_length"))
    onnx_rr = get_reranker(light["retrieval"]["reranker_model"], light["retrieval"].get("rerank_max_length"))
    diffs, same_order = [], 0
    for q in questions[:8]:
        _, cands, _ = pipe.search(q)
        texts = [c.chunk.text for c in cands[:15]]
        x, y = np.array(torch_rr.score(q, texts)), np.array(onnx_rr.score(q, texts))
        diffs.append(float(np.abs(x - y).max()))
        same_order += list(np.argsort(-x)) == list(np.argsort(-y))
    print(f"Reranker: 8 questions x 15 passages | largest score difference = {max(diffs):.6f} | "
          f"same ranking in {same_order}/8")


if __name__ == "__main__":
    main()
