"""Reciprocal rank fusion."""
from __future__ import annotations

from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(chunk) = sum over lists of 1 / (k + rank). Uses ranks only, so no score calibration needed."""
    scores = defaultdict(float)
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
