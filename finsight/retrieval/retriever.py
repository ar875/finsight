"""Hybrid retrieval: dense + BM25, RRF fusion, soft filters, reranking."""
from __future__ import annotations

from ..schema import RetrievedChunk
from .fusion import reciprocal_rank_fusion
from .query_analysis import expand_query

class Retriever:
    def __init__(self, index, embedder, rcfg, reranker=None):
        self.index, self.embedder, self.cfg, self.reranker = index, embedder, rcfg, reranker

    def _search_lists(self, query, qvec, flt):
        mode, k = self.cfg["mode"], self.cfg["top_k"]
        keyword_query = expand_query(query) if self.cfg.get("expand_query") else query
        dense = self.index.dense_search(qvec, k, flt) if mode in ("dense", "hybrid") else []
        sparse = self.index.bm25_search(keyword_query, k, flt) if mode in ("bm25", "hybrid") else []
        return dense, sparse

    def candidates(self, query, hard_filter=None, soft_filter=None) -> list[RetrievedChunk]:
        qvec = self.embedder.embed_query(query)
        dense, sparse = self._search_lists(query, qvec, hard_filter)
        lists = [[c for c, _ in dense], [c for c, _ in sparse]]
        if soft_filter:
            # same search restricted to the preferred year/doc type: matching chunks appear in more lists,
            # so RRF ranks them higher, but other years can still win if they are much more relevant
            d2, s2 = self._search_lists(query, qvec, {**(hard_filter or {}), **soft_filter})
            lists += [[c for c, _ in d2], [c for c, _ in s2]]
        lists = [l for l in lists if l]
        if self.cfg["mode"] == "hybrid" or soft_filter:
            ranked = reciprocal_rank_fusion(lists, self.cfg["rrf_k"])
        else:
            ranked = dense or sparse
        dense_rank = {c: r for r, (c, _) in enumerate(dense, 1)}
        bm25_rank = {c: r for r, (c, _) in enumerate(sparse, 1)}
        return [RetrievedChunk(self.index.chunks[c], s, dense_rank.get(c), bm25_rank.get(c))
                for c, s in ranked[:self.cfg["top_k"]]]

    def retrieve(self, query, hard_filter=None, soft_filter=None, final_k=None):
        """Returns (final chunks for the LLM, all candidates before reranking)."""
        cands = self.candidates(query, hard_filter, soft_filter)
        if self.reranker and self.cfg["use_reranker"] and cands:
            # rerank_top_n: re-rank only the best n fused candidates (faster on CPU); None = all of them
            n = self.cfg.get("rerank_top_n") or len(cands)
            head, tail = cands[:n], cands[n:]
            for c, s in zip(head, self.reranker.score(query, [c.chunk.text for c in head])):
                c.rerank_score = float(s)
            cands = sorted(head, key=lambda c: -c.rerank_score) + tail
        return cands[:final_k or self.cfg["final_k"]], cands
