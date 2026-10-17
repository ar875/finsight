"""Core tests (no model downloads, no API key): fake embedder, keyword reranker, fake LLM.

    pytest -q
"""
from __future__ import annotations

import pymupdf
import pytest

from finsight import RAGPipeline, deep_update
from finsight.generation.llm import LLMResponse
from finsight.generation.prompts import extract_citations, is_refusal
from finsight.indexing.index import tokenize
from finsight.indexing.ingest import ingest
from finsight.ingestion.chunker import _is_real_table, chunk_elements, flatten_table_header
from finsight.ingestion.parser import _is_heading, parse_pdf, rows_to_markdown
from finsight.ingestion.units import detect_page_units
from finsight.retrieval.fusion import reciprocal_rank_fusion
from finsight.retrieval.query_analysis import analyse_question, detect_companies, detect_years, expand_query
from finsight.schema import Element


class FakeLLM:
    def chat(self, messages):
        return LLMResponse("ACME revenue was ₹12,450 crore [1]. Also see [9].", 100, 10)


def _make_test_pdf(path):
    """A small PDF with a repeated header, headings, body text, and one real table."""
    doc = pymupdf.open()
    for p in range(6):
        page = doc.new_page()
        page.insert_text((50, 30), "ACME Annual Report 2024-25", fontsize=8)          # repeated header
        page.insert_text((50, 120), f"Section {p + 1} Overview", fontsize=16)        # heading
        page.insert_text((50, 150), f"Revenue on page {p + 1} grew strongly this year.", fontsize=10)
        if p == 2:
            data = [["Metric", "FY2025", "FY2024"], ["Revenue", "12,450", "11,506"], ["Net profit", "2,310", "2,105"]]
            for r, row in enumerate(data):
                for c, val in enumerate(row):
                    rect = pymupdf.Rect(50 + c * 150, 200 + r * 20, 50 + (c + 1) * 150, 200 + (r + 1) * 20)
                    page.draw_rect(rect, width=0.8)
                    page.insert_text((rect.x0 + 4, rect.y1 - 6), val, fontsize=10)
        page.insert_text((300, 820), str(p + 1), fontsize=8)                       # page number
    doc.save(path)


def test_parser_tables_headings_headers(tmp_path):
    _make_test_pdf(tmp_path / "test.pdf")
    els = parse_pdf(tmp_path / "test.pdf")
    assert any(e.kind == "table" and "| Revenue | 12,450 | 11,506 |" in e.text for e in els), "table not found"
    assert any(e.kind == "heading" and e.text == "Section 1 Overview" for e in els), "heading not found"
    assert not any("ACME Annual Report 2024-25" in e.text for e in els), "header not removed"

    chunks = chunk_elements(els, "ACME_FY2025_annual_report", {"title": "ACME FY2025 Annual Report"})
    table = [c for c in chunks if c.content_type == "table"][0]
    assert table.text.startswith("[ACME FY2025 Annual Report | Section 3 Overview]") and table.page_start == 3


def test_chunking():
    big = [["Metric", "Value"]] + [[f"row{i}", str(i)] for i in range(300)]
    pieces = chunk_elements([Element(1, "heading", "Big"), Element(1, "table", rows_to_markdown(big))], "D", {}, "structure", 120, 0)
    assert len(pieces) > 3 and all("| Metric | Value |" in c.text for c in pieces)
    fixed = chunk_elements([Element(1, "text", "a " * 600), Element(2, "text", "b " * 600)], "D", {}, "fixed", 256, 32)
    assert fixed[0].page_start == 1 and fixed[-1].page_end == 2
    tiny = [Element(1, "heading", "A"), Element(1, "text", "short text."), Element(1, "heading", "B"),
            Element(1, "text", "more short text."), Element(2, "heading", "C"), Element(2, "text", "word " * 300)]
    assert len(chunk_elements(tiny, "D", {"title": "T"}, "structure", 512, 0, min_section_tokens=200)) == 1
    assert not _is_real_table("| 754 |  | 458 |  | 893 |\n| --- | --- | --- | --- | --- |\n| 191, |  | 225, |  | 240, |")
    assert _is_real_table("|  | 2025 | 2024 |\n| --- | --- | --- |\n| Revenue | 12,450 | 11,506 |\n| Net profit | 2,310 | 2,105 |")
    box = [Element(1, "heading", "H"), Element(1, "table", "| Note |\n| --- |\n| See page 4 |"), Element(1, "text", "x " * 300)]
    assert all(c.content_type == "text" for c in chunk_elements(box, "D", {}, "structure", 512, 0))


def test_question_understanding():
    q = analyse_question("Compare HDFC Bank and ICICI Bank gross NPA as of March 31, 2025")
    assert q["companies"] == ["HDFCBANK", "ICICIBANK"] and q["years"] == ["FY2025"]
    q = analyse_question("What revenue guidance did Infosys give for FY26?")
    assert q["companies"] == ["INFY"] and q["years"] == ["FY2025"] and q["doc_type"] == "earnings_call"
    assert detect_companies("the scheme was hailed by many") == []      # "HUL" must not match inside words
    assert detect_years("results for 2024-25") == ["FY2025"]
    assert tokenize("FY26 guidance") == tokenize("FY 2026 guidance") == tokenize("2025-26 guidance")
    assert "profit after tax" in expand_query("What was the net profit?")
    assert RAGPipeline.sub_question("Compare the net profit of TCS and Wipro in FY2025", "WIPRO") == "Wipro net profit in FY2025"
    assert RAGPipeline.sub_question("What was the gross NPA of HDFC Bank and ICICI Bank as of March 31, 2025?",
                                    "ICICIBANK") == "ICICI Bank gross NPA as of March 31, 2025"


@pytest.mark.parametrize("bad", ["Gross NPA Ratio (%)", "CAGR 10.2%", "March 31, 2025", "Total", "Subject",
                                 "Aparna Iyer:", "12,450 3,210"])
def test_not_headings(bad):
    assert not _is_heading(bad, 14, True, 10)


def test_headings():
    assert _is_heading("FINANCIAL PERFORMANCE", 10, True, 10) and _is_heading("Risk Management", 14, False, 10)
    assert not _is_heading("Jayesh Sanghrajka", 10, True, 10, allow_bold=False)     # transcript speaker name


def test_units_and_flat_headers():
    assert detect_page_units("CONSOLIDATED CASH FLOW STATEMENT  ₹ in ‘000s  Year ended") == "₹ thousand"
    assert detect_page_units("( C in crore) As at March 31, 2026") == "₹ crore"
    assert detect_page_units("( I in millions, except share data") == "₹ million"
    assert detect_page_units("bought back shares for ₹ 18,000 Crore") is None
    two_row = "|  | FY 2026 |  |\n| --- | --- | --- |\n|  | Male % | Female % |\n| Permanent | 13.8 | 13.4 |"
    assert flatten_table_header(two_row).split("\n")[0] == "|  | FY 2026 Male % | FY 2026 Female % |"
    plain = "|  | 2025 | 2024 |\n| --- | --- | --- |\n| Revenue | 12,450 | 11,506 |\n| Profit | 2,310 | 2,105 |"
    assert flatten_table_header(plain) == plain
    unit_chunks = chunk_elements([Element(3, "heading", "Schedules"), Element(3, "text", "word " * 120)], "D",
                                 {"title": "T"}, page_units={3: "₹ thousand"})
    assert unit_chunks[0].text.startswith("[T | Schedules | amounts in ₹ thousand]")


def test_citations_and_refusals():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])[0][0] == "b"
    assert extract_citations("NPA was 1.33% 【5】 and 【2†L10】.") == [5, 2]
    assert is_refusal("NOT_FOUND: not in sources")
    assert is_refusal("The call does not say. NOT_FOUND: not mentioned.")      # mid-answer refusal (g100)
    assert not is_refusal("Revenue was ₹2,55,324 crore [1].")


def test_ingest_and_pipeline(tiny_cfg):
    stats = ingest(tiny_cfg, log=lambda *a: None)
    assert stats["added"] == 3 and stats["duplicates_dropped"] == 1 and not stats["unmatched_filenames"]
    assert ingest(tiny_cfg, log=lambda *a: None)["skipped"] == 3                 # incremental: nothing re-processed

    pipe = RAGPipeline(tiny_cfg, llm=FakeLLM())
    final, _, info = pipe.search("What was TCS revenue from operations in FY2025?")
    assert info["hard_filter"] == {"company": ["TCS"]} and final[0].chunk.doc_id == "TCS_FY2025_annual_report"
    final, _, _ = pipe.search("Compare TCS revenue and HDFC Bank gross NPA")
    assert {c.chunk.metadata["company"] for c in final} == {"TCS", "HDFCBANK"}
    result = pipe.ask("What was TCS revenue in FY2025?")
    assert [c["number"] for c in result["citations"]] == [1]                     # [9] does not exist -> dropped
    assert not result["refused"] and not result["gated"]


def test_relevance_gate(tiny_cfg):
    ingest(tiny_cfg, log=lambda *a: None)
    gated_cfg = deep_update(tiny_cfg, {"generation": {"min_rerank_score": 0.99}})

    class MustNotBeCalled:
        def chat(self, messages):
            raise AssertionError("the LLM must not be called when the gate refuses")

    r = RAGPipeline(gated_cfg, llm=MustNotBeCalled()).ask("What was Wipro's revenue in FY2010?")
    assert r["refused"] and r["gated"] and r["answer"].startswith("NOT_FOUND")


def test_profiles_and_env_overrides(monkeypatch):
    from finsight import load_config
    full, light = load_config(), load_config(profile="light")
    assert full["retrieval"]["reranker_model"] == "BAAI/bge-reranker-base" and "profile" not in full
    assert light["retrieval"]["reranker_model"] != full["retrieval"]["reranker_model"] and light["profile"] == "light"
    assert light["chunking"] == full["chunking"]                     # same index for both profiles
    monkeypatch.setenv("FINSIGHT_GATE", "0.05")
    assert load_config(profile="light")["generation"]["min_rerank_score"] == 0.05
    with pytest.raises(ValueError):
        load_config(profile="nope")


def test_rerank_top_n(tiny_cfg):
    ingest(tiny_cfg, log=lambda *a: None)
    pipe = RAGPipeline(deep_update(tiny_cfg, {"retrieval": {"rerank_top_n": 1, "company_filter": False}}), llm=FakeLLM())
    final, cands, _ = pipe.search("revenue from operations")
    assert sum(c.rerank_score is not None for c in cands) == 1 and final[0].rerank_score is not None


def test_index_saves_keyword_index(tiny_cfg):
    from finsight.indexing.index import DocumentIndex
    from finsight import index_dir
    ingest(tiny_cfg, log=lambda *a: None)
    folder = index_dir(tiny_cfg)
    assert (folder / "bm25.npz").exists()
    fresh = DocumentIndex(folder, "numpy")                       # a new process would load the saved arrays
    assert fresh.bm25_search("revenue operations", 5) == RAGPipeline(tiny_cfg, llm=FakeLLM()).index.bm25_search("revenue operations", 5)
