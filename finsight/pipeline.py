"""RAG pipeline: question analysis -> hybrid retrieval -> relevance gate -> cited answer."""
from __future__ import annotations

import re
import time

from .companies import COMPANIES
from .generation.llm import llm_from_config
from .generation.prompts import build_messages, extract_citations, is_refusal, normalize_citations
from .indexing.embedder import get_embedder
from .indexing.index import get_index
from .retrieval.query_analysis import analyse_question
from .retrieval.reranker import get_reranker
from .retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, cfg, llm=None):
        self.cfg = cfg
        r = cfg["retrieval"]
        self.embedder = get_embedder(cfg["embedding"]["model"], cfg["embedding"]["batch_size"],
                                     cfg["embedding"].get("onnx_model"))
        self.index = get_index(cfg)
        self.reranker = get_reranker(r["reranker_model"], r.get("rerank_max_length")) if r["use_reranker"] else None
        self.retriever = Retriever(self.index, self.embedder, r, self.reranker)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = llm_from_config(self.cfg["generation"])
        return self._llm

    def search(self, question, filters=None):
        """Returns (final chunks, all candidates, info about filters used)."""
        r = self.cfg["retrieval"]
        q = analyse_question(question)
        hard = dict(filters or {})
        if r["company_filter"] and q["companies"] and "company" not in hard:
            hard["company"] = q["companies"]
        soft = {}
        if r["soft_filters"]:
            if q["years"]:
                soft["year"] = q["years"]
            if q["doc_type"]:
                soft["doc_type"] = q["doc_type"]
        info = {"analysis": q, "hard_filter": hard, "soft_filter": soft}
        companies = hard.get("company") or []
        if r["decompose_multi_company"] and len(companies) > 1:
            # comparison: search each company separately, with a question about that company only
            per = r.get("per_company_k", 3)
            results = [self.retriever.retrieve(self.sub_question(question, c), {**hard, "company": [c]}, soft,
                                               final_k=per) for c in companies]
            final = [res[0][i] for i in range(per) for res in results if i < len(res[0])]
            return final, [c for res in results for c in res[1]], info
        final, cands = self.retriever.retrieve(question, hard or None, soft or None)
        return final, cands, info

    @staticmethod
    def sub_question(question, company):
        """'Compare the net profit of TCS and Wipro in FY2025' -> 'Wipro net profit in FY2025'"""
        q = re.sub(r"\b(compare|comparison of|versus|vs\.?|between|and|both)\b", " ", question, flags=re.I)
        for info in COMPANIES.values():
            for name in sorted({info["name"], *info["aliases"]}, key=len, reverse=True):
                q = re.sub(rf"(?<![\w&]){re.escape(name)}(?:'s|’s)?(?![\w&])", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        q = re.sub(r"\b(of|for)\s+(?=(in|for|of|during|as|at)\b)", "", q)   # "net profit of in FY2025" -> "net profit in FY2025"
        q = re.sub(r"^(the|what (is|was|were) the|what (is|was))\s+", "", q, flags=re.I)
        return f"{COMPANIES[company]['name']} {q}".strip(" ?")

    def ask(self, question, filters=None) -> dict:
        t0 = time.perf_counter()
        chunks, cands, info = self.search(question, filters)
        t1 = time.perf_counter()
        base = {"question": question, "filters": info, "sources": [c.ref() for c in chunks],
                "candidates": [c.ref() for c in cands], "citations": [], "error": None, "tokens": {}}
        top = chunks[0].rerank_score if chunks and chunks[0].rerank_score is not None else None
        if not chunks:
            return {**base, "answer": "NOT_FOUND: no matching documents.", "refused": True, "grounded": True,
                    "gated": True, "latency_ms": {"retrieval": round((t1 - t0) * 1000), "total": round((t1 - t0) * 1000)}}
        if top is not None and top < self.cfg["generation"]["min_rerank_score"]:
            # relevance gate: the best passage is not about the question -> refuse without calling the LLM
            return {**base, "answer": "NOT_FOUND: retrieved passages are not relevant enough.", "refused": True,
                    "grounded": True, "gated": True, "top_rerank_score": top,
                    "latency_ms": {"retrieval": round((t1 - t0) * 1000), "total": round((t1 - t0) * 1000)}}
        out = self.llm.chat(build_messages(question, chunks))
        t2 = time.perf_counter()
        answer, error = normalize_citations(out.text.strip()), None
        if not answer:
            error = f"Empty answer (finish_reason={out.finish_reason}). Increase max_tokens or lower reasoning_effort."
            answer = "Sorry, I could not generate an answer. Please try again."
        refused = is_refusal(answer)
        citations = [] if error else [
            {"number": n, **chunks[n - 1].ref()} for n in extract_citations(answer) if 1 <= n <= len(chunks)]
        return {**base, "answer": answer, "refused": refused, "error": error, "citations": citations,
                "grounded": refused or bool(citations), "gated": False, "top_rerank_score": top,
                "latency_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000),
                               "total": round((t2 - t0) * 1000)},
                "tokens": {"input": out.input_tokens, "output": out.output_tokens}}
