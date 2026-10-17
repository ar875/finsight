"""ONNX reranker plumbing with a tiny hand-built model: tokenising pairs, padding, truncation, batching, sigmoid.
(The real check, int8 bge-reranker-base against the original, runs on Kaggle: evaluation/reranker_compare.py.)"""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

from finsight.retrieval.reranker import get_reranker  # noqa: E402


def _build_fake_model(folder):
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate("revenue tcs fy2025 crore was the profit".split())}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
                                                       special_tokens=[("<s>", 1), ("</s>", 2)])
    tok.save(str(folder / "tokenizer.json"))
    # logit = (number of real tokens) * 0.1 - 2  -> longer pairs score higher, padding must not count
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1])
    nodes = [helper.make_node("Cast", ["attention_mask"], ["m"], to=TensorProto.FLOAT),
             helper.make_node("ReduceSum", ["m", "axes"], ["n"], keepdims=1),
             helper.make_node("Mul", ["n", "w"], ["nw"]), helper.make_node("Sub", ["nw", "b0"], ["logits"])]
    inits = [helper.make_tensor("axes", TensorProto.INT64, [1], [1]), helper.make_tensor("w", TensorProto.FLOAT, [], [0.1]),
             helper.make_tensor("b0", TensorProto.FLOAT, [], [2.0])]
    graph = helper.make_graph(nodes, "fake", [ids, mask], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model.onnx"))


def test_onnx_reranker(tmp_path):
    _build_fake_model(tmp_path)
    r = get_reranker(f"onnx:{tmp_path}:model.onnx")
    texts = ["revenue", "tcs revenue fy2025 was crore", "the"] + ["profit"] * 20          # 23 texts -> 2 batches
    s = r.score("tcs revenue", texts)
    assert len(s) == 23 and all(0 < x < 1 for x in s)
    # pair "tcs revenue" + "revenue": 2 + 1 tokens + 4 specials = 7 -> sigmoid(0.7 - 2)
    assert abs(s[0] - 1 / (1 + np.exp(-(0.7 - 2)))) < 1e-6
    assert s[1] > s[0] > s[2] - 1e-9 and s[3] == s[22]                # padding never changes a score
    long = r.score("tcs", ["revenue " * 2000])[0]                     # truncated to 512 tokens
    assert abs(long - 1 / (1 + np.exp(-(51.2 - 2)))) < 1e-6
    assert r.score("tcs", []) == []


def _build_fake_embedder(folder):
    """hidden[b, s, :] = token_id(b, s) * [1, 2, 3, 4]; the embedder takes the first (CLS) position and normalises."""
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "tcs": 4, "revenue": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok.save(str(folder / "tokenizer.json"))
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])
    types = helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, ["b", "s"])
    out = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", 4])
    nodes = [helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT),
             helper.make_node("Unsqueeze", ["f", "ax"], ["f3"]),
             helper.make_node("Mul", ["f3", "w"], ["last_hidden_state"])]
    inits = [helper.make_tensor("ax", TensorProto.INT64, [1], [2]), helper.make_tensor("w", TensorProto.FLOAT, [4], [1, 2, 3, 4])]
    model = helper.make_model(helper.make_graph(nodes, "emb", [ids, mask, types], [out], initializer=inits),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model.onnx"))


def test_onnx_embedder(tmp_path):
    from finsight.indexing.embedder import get_embedder
    _build_fake_embedder(tmp_path)
    emb = get_embedder("BAAI/bge-small-en-v1.5", 64, f"{tmp_path}:model.onnx")
    assert emb.query_prefix.startswith("Represent this sentence")          # bge query prefix kept
    q = emb.embed_query("tcs revenue")
    docs = emb.embed_documents(["tcs"] * 40 + ["revenue tcs tcs"])        # 41 texts -> 2 batches, padding
    expected = np.array([1, 2, 3, 4], np.float32) / np.linalg.norm([1, 2, 3, 4])
    assert q.shape == (4,) and np.allclose(q, expected) and docs.shape == (41, 4) and np.allclose(docs, expected)
    assert emb.embed_documents([]).shape == (0, 4)
