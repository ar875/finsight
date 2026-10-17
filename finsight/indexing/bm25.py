"""BM25 keyword search with the same scores as rank_bm25.BM25Okapi, in a fraction of the memory.

rank_bm25 keeps one Python dict per passage (21K dicts, hundreds of MB). This version keeps one sorted array of
(passage, term count) postings per term, packed into three flat numpy arrays. The arithmetic in get_scores is the
same expression, in the same order, so scores are bit-identical (checked in tests/test_bm25.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        postings: dict[str, list[int]] = {}          # term -> [doc, tf, doc, tf, ...] (first-seen order, like rank_bm25)
        doc_len, total = [], 0
        for i, doc in enumerate(corpus):
            doc_len.append(len(doc))
            total += len(doc)
            counts: dict[str, int] = {}
            for word in doc:
                counts[word] = counts.get(word, 0) + 1
            for word, c in counts.items():
                postings.setdefault(word, []).extend((i, c))
        self.corpus_size = len(doc_len)
        if not self.corpus_size:
            raise ValueError("BM25 needs at least one passage: the index is empty (build it with scripts/build_index.py)")
        self.avgdl = total / self.corpus_size
        self.doc_len = np.array(doc_len)

        # idf exactly as BM25Okapi._calc_idf (same iteration order -> same average_idf)
        self.idf, idf_sum, negative = {}, 0, []
        for word, plist in postings.items():
            freq = len(plist) // 2
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative.append(word)
        self.average_idf = idf_sum / len(self.idf)
        for word in negative:
            self.idf[word] = self.epsilon * self.average_idf

        # pack postings into flat arrays
        self.term_index = {w: n for n, w in enumerate(postings)}
        sizes = np.array([len(p) // 2 for p in postings.values()], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        flat = np.fromiter((x for p in postings.values() for x in p), dtype=np.int64, count=int(sizes.sum()) * 2)
        self.doc_ids = flat[0::2].astype(np.int32)
        self.tfs = flat[1::2].astype(np.int32)
        self._norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    def _freqs(self, term: str) -> np.ndarray:
        q_freq = np.zeros(self.corpus_size, dtype=np.int64)
        n = self.term_index.get(term)
        if n is not None:
            lo, hi = self.offsets[n], self.offsets[n + 1]
            q_freq[self.doc_ids[lo:hi]] = self.tfs[lo:hi]
        return q_freq

    def get_scores(self, query: list[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        for q in query:
            q_freq = self._freqs(q)
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) / (q_freq + self._norm))
        return score

    # ---- save / load: the website loads these arrays instead of re-tokenising 21K passages at start-up
    def save(self, path) -> None:
        terms = list(self.term_index)
        np.savez(path, doc_ids=self.doc_ids, tfs=self.tfs, offsets=self.offsets, doc_len=self.doc_len,
                 idf=np.array([self.idf[t] for t in terms], dtype=np.float64),
                 params=np.array([self.k1, self.b, self.epsilon, self.avgdl, self.average_idf], dtype=np.float64))
        Path(path).with_suffix(".terms.json").write_text(json.dumps(terms, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "BM25":
        data = np.load(path)
        terms = json.loads(Path(path).with_suffix(".terms.json").read_text())
        self = cls.__new__(cls)
        self.k1, self.b, self.epsilon, self.avgdl, self.average_idf = (float(x) for x in data["params"])
        self.doc_ids, self.tfs, self.offsets, self.doc_len = data["doc_ids"], data["tfs"], data["offsets"], data["doc_len"]
        self.corpus_size = len(self.doc_len)
        self.term_index = {t: n for n, t in enumerate(terms)}
        self.idf = dict(zip(terms, data["idf"].tolist()))
        self._norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        return self
