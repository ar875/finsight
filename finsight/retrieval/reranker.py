"""Cross-encoder reranker (real) and keyword reranker (tests only)."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from ..indexing.index import tokenize


class CrossEncoderReranker:
    """Reads question and chunk TOGETHER, so it judges relevance better than embeddings. Slower, so only top ~30."""

    def __init__(self, model_name, max_length=None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length) if max_length else CrossEncoder(model_name)

    def score(self, query, texts):
        if not texts:
            return []
        s = np.asarray(self.model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False), float)
        if s.min() < 0 or s.max() > 1:
            s = 1 / (1 + np.exp(-s))         # logits -> 0..1 so thresholds are meaningful
        return s.tolist()


class KeywordReranker:
    """Tiny stand-in for tests only."""
    def score(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


class OnnxCrossEncoderReranker:
    """The same cross-encoder exported to ONNX and run with onnxruntime: no PyTorch, much less memory.
    Model spec in config: "onnx:<hugging face repo or local folder>:<path of the .onnx file in it>"."""

    def __init__(self, repo: str, filename: str, max_length: int | None = None, batch_size: int = 8):
        from ..onnx_utils import load_session, load_tokenizer
        self.tokenizer = load_tokenizer(repo, max_length or 512)
        self.session = load_session(repo, filename)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size

    def score(self, query, texts):
        from ..onnx_utils import encode_feed
        out = []
        for i in range(0, len(texts), self.batch_size):
            feed = encode_feed(self.tokenizer, [(query, t) for t in texts[i:i + self.batch_size]], self.inputs)
            logits = self.session.run(None, feed)[0]
            out.extend(np.asarray(logits, float).reshape(len(feed["input_ids"]), -1)[:, 0])
        return (1 / (1 + np.exp(-np.asarray(out, float)))).tolist()           # logits -> 0..1, same scale as before


@lru_cache(maxsize=2)
def get_reranker(model_name, max_length=None):
    """max_length: tokens per (question, passage) pair; None = the model's own limit (512 for bge-reranker-base)."""
    if model_name == "keyword":
        return KeywordReranker()
    if model_name.startswith("onnx:"):
        from ..onnx_utils import parse_spec
        return OnnxCrossEncoderReranker(*parse_spec(model_name), max_length=max_length)
    return CrossEncoderReranker(model_name, max_length)
