"""Scorer v2 cases taken from real Phase 3/4 answers."""
from evaluation.scoring import score_answer, score_numeric


def test_unit_conversion_and_number_words():
    assert score_numeric("₹16,103,480,210 thousand", "total deposits were ₹16,103.48 billion [2]")["correct"] == 1.0
    assert score_numeric("at least 4 months", "unchanged for the last four months [3]")["correct"] == 1.0
    assert score_numeric("₹69,952.7 million", "The additional investment was ₹6,995.27 crore [1]")["correct"] == 1.0


def test_missing_scale_and_currency_errors():
    # g045: the answer dropped "thousand" -> wrong by 1,000x
    assert score_numeric("₹1,202,409,132 thousand", "was ₹1,202,409,132 [1]")["correct"] == 0.0
    # m003: right digits, US$ instead of ₹
    s = score_numeric("ICICI Bank: ₹16,103,480,210 thousand", "ICICI Bank deposits were US$16,103.48 billion [8]")
    assert s["currency_error"] and s["correct"] == 0.0


def test_partial_and_wrong():
    assert score_numeric("₹2,55,324 crore and 24.6%", "Revenue was ₹2,55,324 crore [1]")["correct"] == 0.5
    assert score_numeric("₹48,553 crore", "TCS's net profit was ₹46,099 crore [1]")["correct"] == 0.0


def _result(answer, refused=False):
    return {"answer": answer, "refused": refused, "grounded": True, "citations": [], "sources": [], "candidates": []}


def test_accept_also_and_refusals():
    q = {"id": "g131", "type": "table", "question": "HUL cash?", "answer": "₹5,947 crore", "evidence": []}
    r = _result("Hindustan Unilever's cash and cash equivalents were ₹6,071 crore [1].")
    assert score_answer(q, r)["correct"] == 0.0
    s = score_answer(q, r, accept_also={"g131": "₹6,071 crore"})
    assert s["correct"] == 1.0 and s["accepted_alternative"]
    u = {"id": "u005", "type": "unanswerable", "question": "HUL revenue FY2014?", "answer": "", "evidence": []}
    assert score_answer(u, _result("NOT_FOUND: no data", refused=True))["correct"] == 1.0
    assert score_answer(u, _result("₹59,579 crore [1]"))["correct"] == 0.0
