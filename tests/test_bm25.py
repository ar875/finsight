"""The compact BM25 must give exactly the same scores as rank_bm25 (which the evaluation used)."""
import random

import numpy as np
import pytest

from finsight.indexing.bm25 import BM25

rank_bm25 = pytest.importorskip("rank_bm25")


def test_identical_to_rank_bm25():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(3000)] + ["fy2025", "revenue", "crore", "the"]
    weights = [1 / (i + 1) for i in range(len(vocab))]                      # Zipf-like: some very common terms
    corpus = [rng.choices(vocab, weights, k=rng.randint(1, 300)) for _ in range(2000)] + [["_"]]
    ours, ref = BM25(corpus), rank_bm25.BM25Okapi(corpus)
    assert ours.average_idf == ref.average_idf and ours.idf == ref.idf
    for _ in range(50):
        query = rng.choices(vocab, k=rng.randint(1, 12)) + ["not_in_corpus"]
        assert np.array_equal(ours.get_scores(query), ref.get_scores(query))   # bit-identical, not just close


def test_save_load_identical(tmp_path):
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(500)] + ["₹", "crore"]
    corpus = [rng.choices(vocab, k=rng.randint(1, 80)) for _ in range(300)]
    built = BM25(corpus)
    built.save(tmp_path / "bm25.npz")
    loaded = BM25.load(tmp_path / "bm25.npz")
    for _ in range(20):
        q = rng.choices(vocab, k=5) + ["missing"]
        assert np.array_equal(built.get_scores(q), loaded.get_scores(q))
