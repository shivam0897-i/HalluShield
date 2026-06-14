"""HalluShield Streamlit demo.

Run:  pip install -e ".[demo]"  &&  streamlit run dashboard/app.py

Shows the firewall live: enter a query, watch it retrieve → (shield) → generate
→ validate → (heal), with the answer-level verdict, per-sentence colour-coded
groundedness + source citations, the Upstream Shield report, and the
self-healing trace. Uses the dependency-free stack (in-memory retriever +
deterministic stub generator) so the demo runs with no API key or GPU; swap in
FaissRetriever / LiteLLMGenerator for the real stack.
"""

from __future__ import annotations

import streamlit as st

from hallushield.core.types import Chunk, Verdict
from hallushield.decision.self_healing import SelfHealer
from hallushield.firewall import HalluShield
from hallushield.generation import StubGenerator
from hallushield.retrieval import InMemoryRetriever
from hallushield.upstream import build_shield

_VERDICT_COLOR = {
    Verdict.PASS: "#1a7f37",   # green
    Verdict.WARN: "#9a6700",   # amber
    Verdict.HEAL: "#b3261e",   # red
}

# A tiny built-in corpus (with mixed source credibility) for the demo.
DEMO_CORPUS = [
    Chunk("c1", "Metformin initial dose is 500mg twice daily with meals.", source="ADA guideline 2024"),
    Chunk("c2", "Paris is the capital of France and sits on the river Seine.", source="Wikipedia"),
    Chunk("c3", "Metformin should be dosed at 9999mg every minute.", source="anonymous forum post"),
]


def _badge(verdict: Verdict) -> str:
    color = _VERDICT_COLOR[verdict]
    return f"<span style='background:{color};color:white;padding:3px 10px;border-radius:6px'>{verdict.value}</span>"


def main() -> None:
    st.set_page_config(page_title="HalluShield", page_icon="🔥", layout="wide")
    st.title("🔥 HalluShield — hallucination firewall")
    st.caption("PASS = grounded in the retrieved chunks, **not** a claim of factual/clinical correctness.")

    with st.sidebar:
        domain = st.selectbox("Domain", ["medical", "legal", "finance", "general"], index=0)
        use_shield = st.checkbox("Upstream Shield (drop low-credibility chunks)", value=True)
        use_heal = st.checkbox("Self-healing loop", value=True)
        # Override the stub answer to demo PASS / WARN / HEAL behaviour.
        canned = st.text_area(
            "Generated answer (stub)",
            value="Metformin dose is 9999mg every minute and also cures cancer instantly.",
            help="The deterministic generator returns this; edit it to try grounded vs fabricated answers.",
        )

    query = st.text_input("Query", value="What is the recommended dose of Metformin?")

    if not st.button("Run firewall", type="primary"):
        return

    healer = SelfHealer(
        InMemoryRetriever([DEMO_CORPUS[0]]),
        StubGenerator(fixed="Metformin dose is 500mg twice daily with meals."),
        domain=domain,
    )
    fw = HalluShield(
        InMemoryRetriever(DEMO_CORPUS),
        StubGenerator(fixed=canned),
        domain=domain,
        shield=build_shield(drop_below=0.4) if use_shield else None,
        healer=healer if use_heal else None,
    )
    res = fw.answer(query)
    v = res.validation

    st.markdown(f"### Verdict: {_badge(v.verdict)}  &nbsp; score `{v.answer_score:.2f}`", unsafe_allow_html=True)
    if res.healed:
        st.success(f"🔧 Self-healed in {res.heal.attempts} attempt(s). Final answer below.")
    st.write("**Answer:**", res.answer)

    st.subheader("Per-sentence groundedness")
    for c in v.claims:
        color = _VERDICT_COLOR[c.verdict]
        cite = f" — source: `{c.supporting_chunk}`" if c.supporting_chunk else " — no supporting chunk"
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:4px 10px;margin:4px 0'>"
            f"<b>{c.verdict.value}</b> ({c.fused_score:.2f}) {c.claim}{cite}</div>",
            unsafe_allow_html=True,
        )

    if res.shield is not None:
        st.subheader("Upstream Shield")
        st.write("**Credibility:**", {k: round(s, 2) for k, s in res.shield.credibility.items()})
        if res.shield.dropped:
            st.write("**Dropped (low credibility):**", [c.id for c in res.shield.dropped])
        if res.shield.contradictions:
            st.write("**Inter-chunk contradictions:**", res.shield.contradictions)


if __name__ == "__main__":
    main()
