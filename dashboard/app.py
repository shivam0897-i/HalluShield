"""HalluShield Streamlit demo — dynamic validator.

Run:  pip install -e ".[demo]"  &&  streamlit run dashboard/app.py

Nothing here is hard-coded into the result: you type a query, paste your LLM's
answer, edit the retrieved chunks in a table, choose which detection signals to
run, and HalluShield validates *your* input live. The answer-level verdict,
per-sentence colour-coded groundedness + citations, and the Upstream Shield
report are all computed from what you enter.

Signals offered depend on what's installed: `lexical` always; `grounding`
(LettuceDetect) with `.[ml]`; `logic` (LLM-as-judge) with `.[api]` + an API key.
"""

from __future__ import annotations

import importlib.util

import streamlit as st

from hallushield import config
from hallushield.core.fusion import FusionScorer
from hallushield.core.types import Chunk, Verdict
from hallushield.pipeline import validate
from hallushield.signals import build_signal
from hallushield.upstream import build_shield

_VERDICT_COLOR = {
    Verdict.PASS: "#1a7f37",   # green
    Verdict.WARN: "#9a6700",   # amber
    Verdict.HEAL: "#b3261e",   # red
}

# Editable starting point only — change/clear any of it to test your own cases.
_EXAMPLE = {
    "query": "What is the recommended dose of metformin?",
    "answer": "Metformin is started at 500mg twice daily with meals. It also instantly cures cancer.",
    "chunks": [
        {"id": "c1", "text": "Metformin initial dose is 500mg twice daily with meals.", "source": "ADA guideline 2024"},
        {"id": "c2", "text": "Metformin should be dosed at 9999mg every minute.", "source": "anonymous forum post"},
    ],
}


def _available_signals() -> list[str]:
    sigs = ["lexical"]
    if importlib.util.find_spec("lettucedetect"):
        sigs.append("grounding")
    if importlib.util.find_spec("litellm"):
        sigs.append("logic")
    return sigs


def _badge(verdict: Verdict) -> str:
    color = _VERDICT_COLOR[verdict]
    return (
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:6px;font-weight:600'>{verdict.value}</span>"
    )


def main() -> None:
    st.set_page_config(page_title="HalluShield", page_icon="🔥", layout="wide")
    st.title("🔥 HalluShield — hallucination firewall")
    st.caption("`PASS` = grounded in the retrieved chunks, **not** a claim of factual/clinical correctness.")

    available = _available_signals()
    with st.sidebar:
        st.header("Settings")
        domain = st.selectbox("Domain (threshold set)", ["medical", "legal", "finance", "general"], index=0)
        signals = st.multiselect("Detection signals", available, default=available)
        st.caption("`grounding` needs `.[ml]`; `logic` needs `.[api]` + an API key. Only installed ones appear.")
        use_shield = st.checkbox("Run Upstream Shield on the chunks", value=True)
        drop_below = st.slider("Shield: drop chunks below credibility", 0.0, 1.0, 0.4, 0.05)

    st.write("Enter a **query**, your model's **answer**, and the retrieved **chunks** (editable). "
             "HalluShield validates the answer against the chunks.")

    query = st.text_input("Query", value=_EXAMPLE["query"])
    answer = st.text_area("Answer (your LLM output)", value=_EXAMPLE["answer"], height=110)
    st.markdown("**Retrieved chunks** — edit text/source, add or delete rows:")
    edited = st.data_editor(
        _EXAMPLE["chunks"], num_rows="dynamic", width="stretch", key="chunks",
        column_config={
            "id": st.column_config.TextColumn("id", required=True),
            "text": st.column_config.TextColumn("text", width="large", required=True),
            "source": st.column_config.TextColumn("source"),
        },
    )

    if not st.button("Validate", type="primary"):
        return

    # ---- assemble input from the form ----
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for row in edited:
        text = str(row.get("text") or "").strip()
        cid = str(row.get("id") or "").strip()
        if not text or not cid or cid in seen:
            continue
        seen.add(cid)
        chunks.append(Chunk(id=cid, text=text, source=(str(row.get("source")).strip() or None) if row.get("source") else None))

    if not answer.strip():
        st.warning("Enter an answer to validate.")
        return
    if not signals:
        st.warning("Select at least one detection signal.")
        return

    # ---- build the fusion scorer from the chosen signals ----
    try:
        with st.spinner(f"Loading signals {signals} …"):
            fusion = FusionScorer(
                [build_signal(name) for name in signals],
                weights=config.DEFAULT_WEIGHTS,
                contradiction_penalty=config.CONTRADICTION_PENALTY,
            )
    except ImportError as exc:
        st.error(str(exc))
        return

    # ---- optional Upstream Shield (runs on the entered chunks) ----
    report = None
    scored_chunks = chunks
    if use_shield and chunks:
        report = build_shield(drop_below=drop_below).validate(chunks)
        scored_chunks = report.kept

    # ---- validate ----
    result = validate(answer, scored_chunks, domain, fusion=fusion, query=query)
    v = result.verdict

    st.markdown(f"### Verdict: {_badge(v)} &nbsp; answer score `{result.answer_score:.2f}`",
                unsafe_allow_html=True)

    st.subheader("Per-sentence groundedness")
    for c in result.claims:
        color = _VERDICT_COLOR[c.verdict]
        cite = f" — source `{c.supporting_chunk}`" if c.supporting_chunk else " — no supporting chunk"
        per_sig = "  ".join(f"{n}={r.score:.2f}" for n, r in c.signals.items())
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:5px 10px;margin:4px 0'>"
            f"<b>{c.verdict.value}</b> ({c.fused_score:.2f}) {c.claim}{cite}"
            f"<br><span style='color:#666;font-size:0.85em'>{per_sig}</span></div>",
            unsafe_allow_html=True,
        )

    if report is not None:
        st.subheader("Upstream Shield")
        st.write("**Source credibility:**", {k: round(s, 2) for k, s in report.credibility.items()})
        if report.dropped:
            st.write("**Dropped (below threshold):**", [c.id for c in report.dropped])
        if report.contradictions:
            st.write("**Inter-chunk contradictions:**", report.contradictions)
        if not report.dropped and not report.contradictions:
            st.caption("No chunks dropped and no contradictions detected.")


if __name__ == "__main__":
    main()
