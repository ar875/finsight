"""Embedding models: sentence-transformers (real) and a hash embedder (tests only)."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

class HashEmbedder:
    """Fake embedder for quick tests only. Never report results from it."""
    dim = 384

    def _vector(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts):
        return np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vector(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model, self.batch_size = SentenceTransformer(model_name), batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        lname = model_name.lower()
        self.query_prefix = ("Represent this sentence for searching relevant passages: " if "bge" in lname
                             else "query: " if "e5" in lname else "")
        self.doc_prefix = "passage: " if "e5" in lname else ""

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self.model.encode([self.doc_prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)

    def embed_query(self, text):
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)


def _prefixes(model_name: str) -> tuple[str, str]:
    lname = model_name.lower()
    query = ("Represent this sentence for searching relevant passages: " if "bge" in lname
             else "query: " if "e5" in lname else "")
    return query, "passage: " if "e5" in lname else ""


class OnnxEmbedder:
    """The same bge model as an ONNX file (onnxruntime, no PyTorch): CLS pooling + L2 normalisation,
    exactly what sentence-transformers does for bge models. Used by the light profile to embed questions."""

    def __init__(self, spec: str, model_name: str, batch_size: int = 32, max_length: int = 512):
        from ..onnx_utils import load_session, load_tokenizer, parse_spec
        repo, filename = parse_spec(spec)
        self.tokenizer = load_tokenizer(repo, max_length)
        self.session = load_session(repo, filename)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.query_prefix, self.doc_prefix = _prefixes(model_name)
        self.batch_size = batch_size
        self.dim = None

    def _encode(self, texts):
        from ..onnx_utils import encode_feed
        out = []
        for i in range(0, len(texts), self.batch_size):
            hidden = self.session.run(None, encode_feed(self.tokenizer, texts[i:i + self.batch_size], self.inputs))[0]
            cls = np.asarray(hidden, np.float32)[:, 0, :]
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        vecs = np.vstack(out).astype(np.float32)
        self.dim = vecs.shape[1]
        return vecs

    def embed_documents(self, texts):
        return self._encode([self.doc_prefix + t for t in texts]) if texts else np.zeros((0, self.dim or 0), np.float32)

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]


@lru_cache(maxsize=4)
def get_embedder(model_name, batch_size=64, onnx_model=None):
    """onnx_model: 'repo:file.onnx' to run the same model with onnxruntime instead of PyTorch."""
    if model_name == "hash":
        return HashEmbedder()
    if onnx_model:
        return OnnxEmbedder(onnx_model, model_name)
    return SentenceTransformerEmbedder(model_name, batch_size)
