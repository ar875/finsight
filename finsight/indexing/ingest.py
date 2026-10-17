"""Ingestion: parallel parsing + parse cache + incremental indexing + dedup."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ..config import index_key
from ..ingestion.chunker import chunk_elements
from ..ingestion.metadata import extract_metadata
from ..ingestion.parser import SUPPORTED_SUFFIXES, parse_document
from ..ingestion.units import page_units_for
from ..schema import Element
from ..utils import file_hash
from .embedder import get_embedder
from .index import get_index


def _parse_to_cache(args):
    """Runs in a worker process: parse one file and save the elements as JSON."""
    path, cache_file, detect_tables = args
    t = time.time()
    try:
        elements = parse_document(path, detect_tables)
        cache_file.write_text(json.dumps([asdict(e) for e in elements]))
        return path.name, len(elements), round(time.time() - t, 1), None
    except Exception as exc:
        return path.name, 0, round(time.time() - t, 1), f"{type(exc).__name__}: {exc}"


def parse_all(cfg, files, log=print) -> dict:
    """Parse every file once (in parallel) and cache the result. Returns {path: elements}."""
    cache_dir = Path(cfg["data"]["parsed_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    detect = cfg["parsing"]["detect_tables"]
    tag = f"{'t' if detect else 'n'}{cfg['parser_version']}"
    jobs, cache = [], {}
    for path in files:
        cache_file = cache_dir / f"{path.stem}.{file_hash(path)[:12]}.{tag}.json"
        cache[path] = cache_file
        if not cache_file.exists():
            jobs.append((path, cache_file, detect))
    if jobs:
        log(f"Parsing {len(jobs)} files with {cfg['parsing']['workers']} workers ...")
        start = time.time()
        with ProcessPoolExecutor(max_workers=cfg["parsing"]["workers"]) as pool:
            for name, n, secs, err in pool.map(_parse_to_cache, jobs):
                log(f"  {name:<42} {n:>6} elements  {secs:>6}s" + (f"  FAILED: {err}" if err else ""))
        log(f"Parsing took {(time.time() - start) / 60:.1f} min")
    return {p: [Element(**e) for e in json.loads(f.read_text())] for p, f in cache.items() if f.exists()}


def ingest(cfg, force=False, log=print) -> dict:
    raw_dir = Path(cfg["data"]["raw_dir"])
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No documents found in {raw_dir}")
    index = get_index(cfg)
    embedder = get_embedder(cfg["embedding"]["model"], cfg["embedding"]["batch_size"])
    ch = cfg["chunking"]
    stats = {"index": index_key(cfg), "added": 0, "skipped": 0, "chunks_added": 0, "table_chunks": 0,
             "duplicates_dropped": 0, "failed": [], "unmatched_filenames": []}
    start = time.time()
    todo = [p for p in files if force or index.registry.get(p.stem, {}).get("hash") != file_hash(p)]
    stats["skipped"] = len(files) - len(todo)
    parsed = parse_all(cfg, todo, log) if todo else {}
    known = index.content_hashes
    for path in todo:
        if path not in parsed:
            stats["failed"].append(path.name)
            continue
        meta = extract_metadata(path)
        if not meta["matched_pattern"]:
            stats["unmatched_filenames"].append(path.name)
        units = page_units_for(cfg, path) if ch.get("unit_headers") else {}
        chunks = chunk_elements(parsed[path], meta["doc_id"], meta, ch["strategy"], ch["chunk_size"],
                                ch["chunk_overlap"], ch.get("min_section_tokens", 0),
                                page_units=units, flatten_headers=ch.get("flatten_headers", False))
        if ch["dedup"]:      # same paragraph in many reports (disclaimers etc.) is kept only once
            unique = []
            for c in chunks:
                if c.metadata["content_hash"] in known:
                    stats["duplicates_dropped"] += 1
                else:
                    known.add(c.metadata["content_hash"])
                    unique.append(c)
            chunks = unique
        index.add_document(meta["doc_id"], file_hash(path), chunks, embedder.embed_documents([c.text for c in chunks]))
        stats["added"] += 1
        stats["chunks_added"] += len(chunks)
        stats["table_chunks"] += sum(c.content_type == "table" for c in chunks)
    index.save()
    stats["total_chunks"] = len(index.ids)
    stats["minutes"] = round((time.time() - start) / 60, 1)
    return stats
