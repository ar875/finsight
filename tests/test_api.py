"""API tests with a tiny index, fake embedder/reranker and a fake LLM."""
from fastapi.testclient import TestClient

from app.main import create_app
from finsight import RAGPipeline
from finsight.generation.llm import DailyLimitError, LLMResponse
from finsight.indexing.ingest import ingest


class FakeLLM:
    def chat(self, messages):
        return LLMResponse("Tata Consultancy Services' revenue in FY2025 was ₹2,55,324 crore [1].", 2900, 40)


class OutOfBudget:
    def chat(self, messages):
        raise DailyLimitError("Rate limit reached: tokens per day")


def _client(cfg, llm, llm_ready=True):
    ingest(cfg, log=lambda *a: None)
    return TestClient(create_app(RAGPipeline(cfg, llm=llm), cfg, llm_ready=llm_ready))


def test_meta_and_page(tiny_cfg):
    with _client(tiny_cfg, FakeLLM()) as c:
        m = c.get("/api/meta").json()
        assert m["ready"] and m["chunks"] > 0 and len(m["companies"]) == 6 and m["examples"]
        assert "FinSight" in c.get("/").text


def test_ask_returns_cited_answer_with_source_text(tiny_cfg):
    with _client(tiny_cfg, FakeLLM()) as c:
        r = c.post("/api/ask", json={"question": "What was TCS revenue from operations in FY2025?"}).json()
        assert "2,55,324" in r["answer"] and not r["refused"]
        assert r["sources"][0]["cited"] and "2,55,324" in r["sources"][0]["text"]
        assert r["sources"][0]["url"].endswith("#page=1")


def test_company_filter_and_validation(tiny_cfg):
    with _client(tiny_cfg, FakeLLM()) as c:
        r = c.post("/api/search", json={"question": "revenue", "companies": ["HDFCBANK"]}).json()
        assert {s["doc_id"].split("_")[0] for s in r["sources"]} == {"HDFCBANK"}
        assert c.post("/api/ask", json={"question": "x"}).status_code == 422


def test_falls_back_to_passages(tiny_cfg):
    with _client(tiny_cfg, OutOfBudget()) as c:
        r = c.post("/api/ask", json={"question": "What was TCS revenue in FY2025?"}).json()
        assert r["answer"] is None and "daily limit" in r["notice"] and r["sources"]
    with _client(tiny_cfg, FakeLLM(), llm_ready=False) as c:
        r = c.post("/api/ask", json={"question": "What was TCS revenue in FY2025?"}).json()
        assert r["answer"] is None and "No answer model" in r["notice"]


def test_rate_limit(tiny_cfg, monkeypatch):
    monkeypatch.setenv("ASKS_PER_MINUTE", "2")
    with _client(tiny_cfg, FakeLLM()) as c:
        codes = [c.post("/api/ask", json={"question": "TCS revenue FY2025?"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
