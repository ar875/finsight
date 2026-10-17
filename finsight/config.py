"""Loads config.yaml, applies environment overrides, and names the index folder."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]          # the project folder
DEFAULT_CONFIG = ROOT / "config.yaml"

# Earlier setups, kept so the evaluation scripts can compare against them
BASELINE_OVERRIDES = {     # Phase 1
    "chunking": {"strategy": "fixed", "dedup": False, "unit_headers": False, "flatten_headers": False},
    "chunker_version": "2.3",
    "retrieval": {"mode": "dense", "use_reranker": False, "company_filter": False,
                  "soft_filters": False, "decompose_multi_company": False, "final_k": 5,
                  "expand_query": False},
}
PHASE2_OVERRIDES = {       # Phase 2 (before unit headers and flattened table headers)
    "chunker_version": "2.3",
    "chunking": {"unit_headers": False, "flatten_headers": False},
}

# Environment variables that override config values (handy for Docker / Hugging Face Spaces)
_ENV = {
    "FINSIGHT_RAW_DIR": ("data", "raw_dir"),
    "FINSIGHT_PARSED_DIR": ("data", "parsed_dir"),
    "FINSIGHT_INDEX_ROOT": ("data", "index_root"),
    "LLM_BASE_URL": ("generation", "base_url"),
    "LLM_MODEL": ("generation", "model"),
    "FINSIGHT_RERANKER": ("retrieval", "reranker_model"),
    "FINSIGHT_GATE": ("generation", "min_rerank_score"),
    "FINSIGHT_RERANK_TOP_N": ("retrieval", "rerank_top_n"),
}
_NUMERIC = {"min_rerank_score": float, "rerank_top_n": int}


def deep_update(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None, profile: str | None = None) -> dict:
    """profile: a named set of changes from the `profiles:` section of config.yaml (e.g. "light" for a small CPU).
    Order: config.yaml -> profile -> environment variables -> overrides."""
    with open(path or os.getenv("FINSIGHT_CONFIG") or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    profiles = cfg.pop("profiles", {}) or {}
    profile = profile or os.getenv("FINSIGHT_PROFILE")
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown profile '{profile}'. Defined in config.yaml: {sorted(profiles)}")
        cfg = deep_update(cfg, profiles[profile])
        cfg["profile"] = profile
    for env, (section, key) in _ENV.items():
        if os.getenv(env):
            value = os.environ[env]
            cfg[section][key] = _NUMERIC[key](value) if key in _NUMERIC else value
    cfg = deep_update(cfg, overrides)
    for key in ("raw_dir", "parsed_dir", "index_root", "companies", "document_links"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = str(p if p.is_absolute() else ROOT / p)
    cfg["parsing"]["workers"] = max(1, min(int(cfg["parsing"]["workers"]), os.cpu_count() or 1))
    return cfg


def index_key(cfg: dict) -> str:
    """Folder name for an index, e.g. structure512o64m200ufd__bge-small-en-v1.5__p2.2c2.4"""
    ch, emb = cfg["chunking"], cfg["embedding"]["model"].split("/")[-1]
    extra = f"m{ch.get('min_section_tokens', 0)}" if ch["strategy"] == "structure" else ""
    extra += ("u" if ch.get("unit_headers") else "") + ("f" if ch.get("flatten_headers") else "")
    return (f"{ch['strategy']}{ch['chunk_size']}o{ch['chunk_overlap']}{extra}{'d' if ch['dedup'] else ''}"
            f"__{emb}__p{cfg['parser_version']}c{cfg.get('chunker_version', '')}")


def index_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["index_root"]) / index_key(cfg)
