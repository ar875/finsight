"""Document index: chunks + vectors (numpy exact search, optional Qdrant) + BM25 keyword index."""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path

import numpy as np

from ..config import index_dir
from ..schema import Chunk
from .bm25 import BM25

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the to was were will with
what which who how did does do this these those their our your than then there""".split())


def normalize_years(text: str) -> str:
    """FY26, FY 2026, FY'26 and 2025-26 all become 'fy2026', so keyword search matches any style."""
    text = re.sub(r"\bFY\s?'?(\d{2})\b", lambda m: f"fy20{m.group(1)}", text, flags=re.I)
    text = re.sub(r"\bFY\s?(\d{4})\b", lambda m: f"fy{m.group(1)}", text, flags=re.I)
    return re.sub(r"\b20(\d{2})\s?[-–]\s?(\d{2})\b", lambda m: f"fy20{m.group(2)}", text)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize_years(text).lower()) if t not in STOPWORDS]


def matches_filter(meta: dict, flt: dict | None) -> bool:
    for key, wanted in (flt or {}).items():
        values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        if str(meta.get(key)) not in {str(v) for v in values}:
            return False
    return True


class DocumentIndex:
    COLLECTION = "chunks"

    def __init__(self, index_dir, backend="numpy"):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.chunks: dict[str, Chunk] = {}
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.registry: dict[str, dict] = {}     # doc_id -> {"hash": file hash, "chunk_ids": [...]}
        self._bm25, self._client = None, None
        self._meta_cache, self._bm25_cache = {}, (None, None)
        self._lock = threading.Lock()
        if (self.dir / "chunks.jsonl").exists():
            with open(self.dir / "chunks.jsonl", encoding="utf-8") as f:
                for line in f:
                    c = Chunk.from_dict(json.loads(line))
                    self.chunks[c.chunk_id] = c
            self.ids = json.loads((self.dir / "ids.json").read_text())
            self.vectors = np.load(self.dir / "vectors.npy")
            self.registry = json.loads((self.dir / "registry.json").read_text())

    # ---- writes
    @property
    def content_hashes(self) -> set:
        return {c.metadata.get("content_hash") for c in self.chunks.values()}

    def add_document(self, doc_id, file_hash, chunks, vectors):
        self.remove_document(doc_id)
        if chunks:
            self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
            for c in chunks:
                self.chunks[c.chunk_id] = c
                self.ids.append(c.chunk_id)
            if self.backend == "qdrant":
                self._qdrant_upsert(chunks, vectors)
        self.registry[doc_id] = {"hash": file_hash, "chunk_ids": [c.chunk_id for c in chunks]}
        self._changed()

    def remove_document(self, doc_id):
        removed = set(self.registry.pop(doc_id, {}).get("chunk_ids", []))
        if not removed:
            return
        keep = [i for i, cid in enumerate(self.ids) if cid not in removed]
        self.vectors, self.ids = self.vectors[keep], [self.ids[i] for i in keep]
        for cid in removed:
            self.chunks.pop(cid, None)
        if self.backend == "qdrant" and self.client().collection_exists(self.COLLECTION):
            from qdrant_client import models
            self.client().delete(self.COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))
        self._changed()

    def _changed(self):
        self._bm25, self._meta_cache, self._bm25_cache = None, {}, (None, None)

    def save(self):
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for cid in self.ids:
                f.write(json.dumps(self.chunks[cid].to_dict(), ensure_ascii=False) + "\n")
        (self.dir / "ids.json").write_text(json.dumps(self.ids))
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "registry.json").write_text(json.dumps(self.registry))
        for f in (self.dir / "bm25.npz", self.dir / "bm25.terms.json"):
            f.unlink(missing_ok=True)        # chunks changed: the keyword index is rebuilt (and saved) on first use
        if self.ids:
            self._get_bm25()

    def warm_up(self):
        """Builds the BM25 index and metadata arrays now instead of on the first question (server start-up)."""
        if not self.ids:
            raise FileNotFoundError(f"The index at {self.dir} is empty: build it with scripts/build_index.py")
        self._get_bm25()
        for key in ("company", "year", "doc_type"):
            self._meta_array(key)

    # ---- search
    def _meta_array(self, key):
        if key not in self._meta_cache:      # one numpy array per metadata field -> fast vectorised filtering
            self._meta_cache[key] = np.array([str(self.chunks[c].metadata.get(key)) for c in self.ids])
        return self._meta_cache[key]

    def _positions(self, flt):
        if not flt:
            return np.arange(len(self.ids))
        mask = np.ones(len(self.ids), dtype=bool)
        for key, wanted in flt.items():
            values = [str(v) for v in (wanted if isinstance(wanted, (list, tuple, set)) else [wanted])]
            mask &= np.isin(self._meta_array(key), values)
        return np.nonzero(mask)[0]

    def dense_search(self, query_vector, k, flt=None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        if self.backend == "qdrant":
            from qdrant_client import models
            conditions = [models.FieldCondition(key=key, match=models.MatchAny(
                any=[str(v) for v in (vals if isinstance(vals, (list, tuple, set)) else [vals])]))
                for key, vals in (flt or {}).items()]
            result = self.client().query_points(self.COLLECTION, query=query_vector.tolist(), limit=k,
                                                query_filter=models.Filter(must=conditions) if conditions else None)
            return [(p.payload["chunk_id"], float(p.score)) for p in result.points]
        pos = self._positions(flt)
        if len(pos) == 0:
            return []
        scores = self.vectors[pos] @ query_vector
        top = np.argsort(-scores)[:k]
        return [(self.ids[pos[i]], float(scores[i])) for i in top]

    def _get_bm25(self):
        with self._lock:                     # build once, even if two requests arrive together
            if self._bm25 is None:
                saved = self.dir / "bm25.npz"
                if saved.exists():           # saved with the index: no re-tokenising at start-up
                    bm25 = BM25.load(saved)
                    if bm25.corpus_size == len(self.ids):
                        self._bm25 = bm25
                if self._bm25 is None:
                    self._bm25 = BM25([tokenize(self.chunks[cid].text) or ["_"] for cid in self.ids])
                    try:
                        self._bm25.save(saved)
                    except OSError:          # read-only folder: fine, just slower next start
                        pass
        return self._bm25

    def bm25_search(self, query, k, flt=None) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not self.ids or not tokens:
            return []
        cached = self._bm25_cache
        if cached[0] != tokens:              # same query searched with different filters: score only once
            cached = (tokens, self._get_bm25().get_scores(tokens))
            self._bm25_cache = cached
        scores = cached[1]                   # local copy: safe when requests run in parallel
        pos = self._positions(flt)
        if len(pos) == 0:
            return []
        sub = scores[pos]
        return [(self.ids[pos[i]], float(sub[i])) for i in np.argsort(-sub)[:k] if sub[i] > 0]

    # ---- qdrant helpers (optional: pip install qdrant-client)
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=str(self.dir / "qdrant"))
        return self._client

    def _qdrant_upsert(self, chunks, vectors):
        from qdrant_client import models
        if not self.client().collection_exists(self.COLLECTION):
            self.client().create_collection(self.COLLECTION, vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE))
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v.tolist(),
                                     payload={**{k: str(val) for k, val in c.metadata.items()}, "chunk_id": c.chunk_id})
                  for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 256):
            self.client().upsert(self.COLLECTION, points=points[i:i + 256])


_OPEN_INDEXES = {}    # Qdrant local mode allows only ONE connection per folder, so we reuse it


def get_index(cfg) -> DocumentIndex:
    path = str(index_dir(cfg))
    if path not in _OPEN_INDEXES:
        _OPEN_INDEXES[path] = DocumentIndex(path, cfg["index"]["backend"])
    return _OPEN_INDEXES[path]
