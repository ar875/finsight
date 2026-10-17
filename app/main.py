"""FinSight web API + UI.

    uvicorn app.main:app --port 7860          # then open http://localhost:7860

Environment:
    LLM_API_KEY         Groq (or any OpenAI-compatible) key. Without it the app still searches and shows passages.
    LLM_MODEL           default from config.yaml (openai/gpt-oss-20b)
    ASKS_PER_MINUTE     per visitor (default 5)
    ASKS_PER_DAY        for the whole server, protects the free LLM budget (default 60)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from finsight import RAGPipeline, deep_update, index_dir, load_config
from finsight.companies import COMPANIES
from finsight.generation.llm import DailyLimitError
from finsight.web import EVALUATION, EXAMPLES, RateLimiter, document_links

log = logging.getLogger("finsight")
STATIC = Path(__file__).parent / "static"

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=400)
    companies: list[str] | None = None



def create_app(pipeline: RAGPipeline | None = None, cfg: dict | None = None, llm_ready: bool | None = None) -> FastAPI:
    state: dict = {"pipeline": pipeline, "error": None}
    # a visitor should not wait minutes: one quick retry, then fall back to showing the passages
    cfg = deep_update(cfg or load_config(), {"generation": {"max_retries": 1, "timeout": 45}})
    links = document_links(cfg["data"]["document_links"])
    limiter = RateLimiter(int(os.getenv("ASKS_PER_MINUTE", 5)), int(os.getenv("ASKS_PER_DAY", 60)))
    run_lock = threading.Lock()          # one question at a time: the free CPU has 2 cores
    has_llm = bool(os.getenv("LLM_API_KEY")) if llm_ready is None else llm_ready

    @asynccontextmanager
    async def lifespan(_app):
        if state["pipeline"] is None:
            try:
                if not (index_dir(cfg) / "chunks.jsonl").exists():
                    raise FileNotFoundError(f"No index at {index_dir(cfg)}. Build it with scripts/build_index.py "
                                            "or copy the index folder from Kaggle.")
                t = time.time()
                pipe = RAGPipeline(cfg)
                pipe.index.warm_up()
                pipe.search("TCS revenue FY2025")          # loads the embedding model and the reranker now
                state["pipeline"] = pipe
                log.warning("FinSight ready: %d chunks in %.0fs", len(pipe.index.ids), time.time() - t)
            except Exception as exc:                       # keep serving the page, show what is wrong
                state["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("Start-up failed")
        yield

    app = FastAPI(title="FinSight", version="1.0.0", lifespan=lifespan)

    def pipe() -> RAGPipeline:
        if state["pipeline"] is None:
            raise HTTPException(503, state["error"] or "The search index is still loading. Try again in a minute.")
        return state["pipeline"]

    def shape(result: dict) -> dict:
        p = state["pipeline"]
        cited = {c["number"] for c in result.get("citations", [])}
        sources = []
        for i, s in enumerate(result["sources"], 1):
            url = links.get(s["doc_id"])
            sources.append({**s, "number": i, "cited": i in cited, "text": p.index.chunks[s["chunk_id"]].text,
                            "url": f"{url}#page={s['page_start']}" if url else None})
        return {"question": result["question"], "answer": result.get("answer"), "refused": result.get("refused", False),
                "gated": result.get("gated", False), "error": result.get("error"), "sources": sources,
                "candidates_searched": len(result.get("candidates", [])), "filters": result["filters"],
                "top_rerank_score": result.get("top_rerank_score"), "latency_ms": result.get("latency_ms", {}),
                "tokens": result.get("tokens", {})}

    def filters_for(req: AskRequest) -> dict | None:
        codes = [c for c in (req.companies or []) if c in COMPANIES]
        return {"company": codes} if codes else None

    def search_only(req: AskRequest, note: str) -> dict:
        t = time.perf_counter()
        with run_lock:
            final, cands, info = pipe().search(req.question, filters_for(req))
        ms = round((time.perf_counter() - t) * 1000)
        result = {"question": req.question, "answer": None, "sources": [c.ref() for c in final],
                  "candidates": [c.ref() for c in cands], "filters": info, "latency_ms": {"retrieval": ms, "total": ms},
                  "top_rerank_score": final[0].rerank_score if final else None}
        return {**shape(result), "notice": note}

    @app.get("/api/meta")
    def meta():
        p = state["pipeline"]
        docs = sorted({c.doc_id for c in p.index.chunks.values()}) if p else []
        return {"ready": p is not None, "error": state["error"], "llm_ready": has_llm,
                "chunks": len(p.index.ids) if p else 0, "documents": len(docs),
                "companies": [{"code": k, "name": v["name"], "sector": v["sector"]} for k, v in COMPANIES.items()],
                "examples": EXAMPLES, "evaluation": EVALUATION, "model": cfg["generation"]["model"]}

    @app.post("/api/search")
    def search(req: AskRequest):
        return search_only(req, "Passages only: no answer was generated.")

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request):
        pipe()
        if not has_llm:
            return search_only(req, "No answer model is set up on this server, so only the passages are shown.")
        visitor = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
        limited = limiter.check(visitor)
        if limited:
            if "minute" in limited:
                raise HTTPException(429, limited)
            return search_only(req, limited)
        try:
            with run_lock:
                result = pipe().ask(req.question, filters_for(req))
        except DailyLimitError:
            return search_only(req, "The answer model's daily limit is reached. Passages are still shown below.")
        except Exception as exc:
            log.exception("ask failed")
            return search_only(req, f"The answer model did not respond ({type(exc).__name__}). Passages are shown below.")
        return shape(result)

    @app.get("/healthz")
    def health():
        return {"ok": state["pipeline"] is not None, "error": state["error"]}

    @app.get("/")
    def index_page():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
