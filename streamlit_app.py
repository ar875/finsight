"""FinSight on Streamlit Community Cloud.

    streamlit run streamlit_app.py

Secrets (Streamlit Cloud: App settings -> Secrets, or .streamlit/secrets.toml locally):
    LLM_API_KEY = "gsk_..."        # Groq key. Without it the app still searches and shows passages.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid

import streamlit as st

st.set_page_config(page_title="FinSight", page_icon="📑", layout="wide")

try:                                          # Streamlit secrets -> environment, before the pipeline is built
    for _key in st.secrets:                   # LLM_API_KEY, and optional FINSIGHT_* settings (e.g. FINSIGHT_RERANK_TOP_N)
        if (_key.startswith("LLM_") or _key.startswith("FINSIGHT_")) and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:                             # no secrets file: fine, search-only mode
    pass

from finsight import RAGPipeline, deep_update, index_dir, load_config  # noqa: E402
from finsight.generation.llm import DailyLimitError  # noqa: E402
from finsight.web import EVALUATION, EVALUATION_LIGHT, EXAMPLE_ANSWERS, EXAMPLES, RateLimiter, document_links  # noqa: E402

SHORT_NAMES = {"TCS": "TCS", "INFY": "Infosys", "WIPRO": "Wipro", "HDFCBANK": "HDFC Bank",
               "ICICIBANK": "ICICI Bank", "HINDUNILVR": "HUL"}

st.markdown("""
<style>
  .block-container { max-width: 1180px; padding-top: 2.2rem; }
  h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; letter-spacing: -0.02em; }
  .fs-lede { color: #586b66; font-size: 1.05rem; max-width: 62ch; margin-bottom: 0.6rem; }
  .fs-answer { font-family: Georgia, "Times New Roman", serif; font-size: 1.32rem; line-height: 1.5; }
  .fs-answer p { margin: 0 0 0.8rem; }
  .fs-cite { font: 600 0.72rem/1 sans-serif; color: #1f6f5c; border: 1px solid #1f6f5c; border-radius: 3px;
             padding: 2px 4px; margin: 0 2px; vertical-align: 3px; }
  .fs-refused { color: #a8442f; }
  .fs-facts { color: #586b66; font-size: 0.85rem; }
  .fs-facts b { color: inherit; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- shared resources (one copy per server)
@st.cache_resource(show_spinner="Loading 21,321 passages and two models. The first visit takes about a minute.")
def load_pipeline():
    # a visitor should not wait minutes: one quick retry, then fall back to showing the passages
    # "light" profile: small reranker that fits the free 2.7 GB CPU (see README for its measured accuracy)
    cfg = deep_update(load_config(profile=os.getenv("FINSIGHT_PROFILE", "light")),
                      {"generation": {"max_retries": 1, "timeout": 45}})
    if not (index_dir(cfg) / "chunks.jsonl").exists():
        raise FileNotFoundError(f"No index at {index_dir(cfg)}")
    pipe = RAGPipeline(cfg)
    pipe.index.warm_up()
    pipe.search("TCS revenue FY2025")                       # loads the embedding model and the reranker now
    return pipe, cfg, document_links(cfg["data"]["document_links"])


@st.cache_resource
def example_answers() -> dict:
    """Saved answers for the example questions (scripts/precompute_examples.py): instant, no LLM tokens."""
    import json
    try:
        data = json.loads(EXAMPLE_ANSWERS.read_text(encoding="utf-8"))
        pipe, _, _ = load_pipeline()
        answers = data.get("answers", {})
        # use them only if every cited passage still exists in this index
        ok = all(s["chunk_id"] in pipe.index.chunks for a in answers.values() for s in a.get("sources", []))
        order = [q for q in data.get("order", list(answers)) if q in answers]
        return {"made_on": data.get("made_on", ""), "answers": answers if ok else {}, "order": order if ok else []}
    except (OSError, ValueError):
        return {"made_on": "", "answers": {}, "order": []}


@st.cache_resource
def shared_state():
    return {"limiter": RateLimiter(int(os.getenv("ASKS_PER_MINUTE", 5)), int(os.getenv("ASKS_PER_DAY", 60))),
            "lock": threading.Lock()}                       # one question at a time on the small CPU


# ---------------------------------------------------------------- rendering helpers
def md_escape(text: str) -> str:
    """Streamlit markdown treats $...$ as maths and a leading # as a heading."""
    text = text.replace("\\", "\\\\").replace("$", "\\$").replace("~", "\\~")
    return re.sub(r"(?m)^(\s*)#", r"\1\\#", text)


def render_answer(answer: str, n_sources: int) -> str:
    html = []
    for para in re.split(r"\n{2,}", answer.strip()):
        p = (para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("$", "&#36;")
             .replace("\n", "<br>"))
        p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", lambda m: "".join(
            f'<span class="fs-cite">{n.strip()}</span>' for n in m.group(1).split(",")
            if 1 <= int(n) <= n_sources), p)
        html.append(f"<p>{p}</p>")
    return "".join(html)


def passage_markdown(text: str) -> tuple[str, str]:
    """Chunk text -> (context line, markdown body). Tables keep their rows; text lines become paragraphs."""
    lines = text.split("\n")
    ctx = ""
    if lines and re.fullmatch(r"\[.*\]", lines[0]):
        parts = lines.pop(0)[1:-1].split(" | ")[1:]           # drop the document title (shown in the header)
        if parts and lines and lines[0].strip() == parts[0].strip():
            parts = parts[1:]                                  # section name repeated as the first line
        ctx = ", ".join(parts)
    blocks, table = [], []
    for line in lines:
        if line.strip().startswith("|"):
            table.append(line.strip())
            continue
        if table:
            blocks.append("\n".join(table))
            table = []
        if line.strip():
            blocks.append(md_escape(line.strip()))
    if table:
        blocks.append("\n".join(table))
    return ctx, "\n\n".join(blocks)


def run_question(question: str, companies: list[str]) -> dict:
    saved = example_answers()
    if not companies and question in saved["answers"]:
        r = saved["answers"][question]
        return {**r, "candidates": [None] * r.get("candidates_count", 0),
                "notice": None, "saved_on": saved["made_on"]}
    pipe, cfg, _ = load_pipeline()
    state = shared_state()
    filters = {"company": companies} if companies else None
    note = None
    if not os.getenv("LLM_API_KEY"):
        note = "No answer model is set up on this server, so only the passages are shown."
    else:
        visitor = st.session_state.setdefault("visitor", str(uuid.uuid4()))
        note = state["limiter"].check(visitor)
    if note is None:
        try:
            with state["lock"]:
                return pipe.ask(question, filters)
        except DailyLimitError:
            note = "The answer model's daily limit is reached. Passages are still shown."
        except Exception as exc:
            note = f"The answer model did not respond ({type(exc).__name__}). Passages are shown instead."
    t = time.perf_counter()
    with state["lock"]:
        final, cands, info = pipe.search(question, filters)
    ms = round((time.perf_counter() - t) * 1000)
    return {"question": question, "answer": None, "notice": note, "refused": False, "gated": False,
            "sources": [c.ref() for c in final], "candidates": [c.ref() for c in cands], "citations": [],
            "filters": info, "latency_ms": {"retrieval": ms}, "tokens": {}}


def show_result(r: dict):
    pipe, _, links = load_pipeline()
    cited = {c["number"] for c in r.get("citations", [])}
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.caption(r["question"] + (f"  \n(saved answer from {r['saved_on']}; type your own question for a live one)"
                                    if r.get("saved_on") else ""))
        if r.get("notice"):
            st.warning(r["notice"])
        if r["answer"] is None:
            st.markdown('<div class="fs-answer"><p>Read the passages on the right; the closest matches are '
                        'listed first.</p></div>', unsafe_allow_html=True)
        elif r["refused"]:
            before = r["answer"].split("NOT_FOUND")[0].strip()
            reason = ("No passage in these documents matches the question closely enough, so the model was not asked."
                      if r.get("gated") else (r["answer"].split("NOT_FOUND", 1)[1].lstrip(": ").strip()))
            html = (render_answer(before, len(r["sources"])) if before else "") + "<p>Not in these documents.</p>"
            st.markdown(f'<div class="fs-answer fs-refused">{html}</div>', unsafe_allow_html=True)
            if reason:
                st.caption(reason)
        else:
            st.markdown(f'<div class="fs-answer">{render_answer(r["answer"], len(r["sources"]))}</div>',
                        unsafe_allow_html=True)
        if r.get("error"):
            st.caption(r["error"])
        lat, tok, q = r.get("latency_ms", {}), r.get("tokens", {}), r["filters"]["analysis"]
        facts = []
        if "retrieval" in lat:
            facts.append(f"Search <b>{lat['retrieval'] / 1000:.1f} s</b>")
        if "generation" in lat:
            facts.append(f"Answer <b>{lat['generation'] / 1000:.1f} s</b>")
        if tok.get("input"):
            facts.append(f"<b>{tok['input'] + tok['output']:,}</b> tokens")
        if q["companies"]:
            facts.append(f"Companies <b>{', '.join(q['companies'])}</b>")
        if q["years"]:
            facts.append(f"Year boost <b>{', '.join(q['years'])}</b>")
        st.markdown(f'<div class="fs-facts">{"&emsp;".join(facts)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f"**Sources sent to the model**  \n"
                    f"<span class='fs-facts'>best {len(r['sources'])} of {len(r['candidates'])} candidates; "
                    f"cited ones are open</span>", unsafe_allow_html=True)
        for i, s in enumerate(r["sources"], 1):
            title = s["label"].split(", p")[0]
            pages = f"page {s['page_start']}" if s["page_start"] == s["page_end"] else f"pages {s['page_start']}–{s['page_end']}"
            mark = "✓ cited" if i in cited else ""
            with st.expander(f"[{i}]  {title}, {pages}  {mark}", expanded=i in cited):
                ctx, body = passage_markdown(pipe.index.chunks[s["chunk_id"]].text)
                st.caption(", ".join(x for x in [ctx, f"relevance {s['rerank_score']:.2f}"
                                                 if s.get("rerank_score") is not None else ""] if x))
                st.markdown(body)
                if s["doc_id"] in links:
                    st.markdown(f"[Open {pages} of the PDF]({links[s['doc_id']]}#page={s['page_start']})")


# ---------------------------------------------------------------- page
st.markdown("FinSight")
st.title("Ask the annual reports.")
st.markdown('<p class="fs-lede">Six Indian companies, three years of annual reports and Q4 earnings calls. '
            "Every answer cites the page it came from, and the system says so when the documents don't contain "
            "the answer.</p>", unsafe_allow_html=True)

try:
    load_pipeline()
except Exception as exc:
    st.error(f"The search index could not be loaded: {exc}")
    st.stop()

if "question" not in st.session_state:
    st.session_state.question = ""

with st.form("ask"):
    question = st.text_area("Your question", key="question", max_chars=400, height=90,
                            placeholder="What was HDFC Bank's gross NPA ratio as of March 31, 2025?")
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        companies = st.pills("Limit to", list(SHORT_NAMES), format_func=SHORT_NAMES.get, selection_mode="multi")
    with c2:
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)



def use_example():
    text = st.session_state.example
    if text:
        st.session_state.question = text       # callbacks run before the page is drawn again, so this is allowed
        st.session_state.pending = text
    st.session_state.example = None            # chip acts like a button, not a toggle


st.pills("Or try one of these", example_answers()["order"] or EXAMPLES, key="example", on_change=use_example)

pending = st.session_state.pop("pending", None)
to_ask = pending or (question.strip() if submitted else None)
if to_ask is not None and len(to_ask) < 3:
    st.warning("Questions need 3 to 400 characters.")
elif to_ask:
    with st.spinner("Searching 21,321 passages, re-ranking the best 15 on a small free CPU and writing a cited "
                    "answer. This takes about 15 seconds."):
        st.session_state.result = run_question(to_ask, [] if pending else list(companies or []))

if "result" in st.session_state:
    st.divider()
    show_result(st.session_state.result)

st.divider()
a, b = st.columns([1, 1.1], gap="large")
with a:
    st.subheader("How it was tested")
    st.markdown("The answers were checked against a hand-reviewed test set of real questions, each traced back to a "
                "page in the PDFs. Numbers are scored exactly, with units converted (₹ thousand, lakh, crore, "
                "million, billion), so a missing \"thousand\" counts as wrong.")
    st.markdown("Search combines keyword matching and meaning-based matching, narrows to the companies and years in "
                "the question, and re-ranks the best candidates before six passages are sent to the model. When no "
                "passage matches well, the model is not asked at all.")
    st.markdown("This website runs a lighter version on a small free CPU: a compact English reranker that checks the "
                "best 15 candidates. The full version re-ranks 30 with a larger model on a GPU. Both were measured "
                "on the same test set.")
with b:
    w, f = EVALUATION_LIGHT, EVALUATION
    rows = [("Answer accuracy", f"{w['answer_accuracy']:.1%}", f"{f['answer_accuracy']:.1%}"),
            ("Accuracy on questions about tables", f"{w['table_accuracy']:.1%}", f"{f['table_accuracy']:.1%}"),
            ("Right page in the top 6 passages", f"{w['retrieval_hit_at_6']:.0%}", f"{f['retrieval_hit_at_6']:.0%}"),
            ("Unanswerable questions refused", f"{w['correct_refusals']:.0%}", f"{f['correct_refusals']:.0%}"),
            ("Answers with the wrong currency", str(w["currency_errors"]), str(f["currency_errors"])),
            ("Typical time per question", f"{w['median_seconds']:.0f} s", f"{f['median_seconds']:.1f} s")]
    st.markdown("| Measured on 55 test questions | This website | Full version (GPU) |\n|---|---:|---:|\n" +
                "\n".join(f"| {k} | **{a}** | {c} |" for k, a, c in rows))
st.caption("Answers come only from the listed documents and can still be wrong; check the cited page before relying "
           "on a figure.")
