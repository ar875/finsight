"""Helpers for running Hugging Face models as ONNX files with onnxruntime (no PyTorch needed)."""
from __future__ import annotations

from pathlib import Path


def parse_spec(spec: str) -> tuple[str, str]:
    """'Xenova/bge-small-en-v1.5:onnx/model.onnx' (or with an 'onnx:' prefix) -> (repo or folder, file)."""
    spec = spec[5:] if spec.startswith("onnx:") else spec
    repo, filename = spec.split(":", 1)
    return repo, filename


def fetch(repo: str, name: str) -> str:
    """A file from a local folder, or downloaded (and cached) from the Hugging Face Hub."""
    if Path(repo).is_dir():
        return str(Path(repo) / name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name)


def load_tokenizer(repo: str, max_length: int):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(fetch(repo, "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)                 # longest_first, like the HF tokenizer
    pad = next((t for t in ("<pad>", "[PAD]") if tok.token_to_id(t) is not None), None)
    tok.enable_padding(pad_id=tok.token_to_id(pad) if pad else 0, pad_token=pad or "[PAD]")
    return tok


def load_session(repo: str, filename: str):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = False      # free activation memory after each call instead of keeping a big pool
    options.enable_mem_pattern = False        # (lower peak memory on the small website server)
    return ort.InferenceSession(fetch(repo, filename), options, providers=["CPUExecutionProvider"])


def encode_feed(tokenizer, inputs, session_inputs: set) -> dict:
    """Tokenise texts (or (a, b) pairs) into the arrays the ONNX model expects."""
    import numpy as np
    enc = tokenizer.encode_batch(inputs)
    feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64)}
    return {k: v for k, v in feed.items() if k in session_inputs}
