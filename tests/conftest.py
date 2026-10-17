import pytest

from finsight import load_config


@pytest.fixture()
def tiny_cfg(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "TCS_FY2025_annual_report.txt").write_text(
        "# Financial Highlights\nTCS revenue from operations in FY2025 was 2,55,324 crore.\n\n"
        "# Forward-looking statements\nThis report contains forward-looking statements.")
    (raw / "TCS_FY2024_annual_report.txt").write_text(
        "# Financial Highlights\nTCS revenue from operations in FY2024 was 2,40,893 crore.\n\n"
        "# Forward-looking statements\nThis report contains forward-looking statements.")
    (raw / "HDFCBANK_FY2025_annual_report.txt").write_text("# Asset Quality\nHDFC Bank gross NPA ratio was 1.33 percent.")
    return load_config(overrides={"data": {"raw_dir": str(raw), "parsed_dir": str(tmp_path / "parsed"),
                                           "index_root": str(tmp_path / "index")},
                                  "parsing": {"workers": 2}, "embedding": {"model": "hash"},
                                  "chunking": {"min_section_tokens": 0},
                                  "retrieval": {"reranker_model": "keyword"},
                                  "generation": {"min_rerank_score": 0.0}})
