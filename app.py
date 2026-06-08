"""
app.py — Streamlit triage board.

The human-facing end of the pipeline: reads data/triage.json (clusters
+ labels produced by src.pipeline) and lets a newcomer filter to the
issues that fit them. The board surfaces the cluster — the distinct
problem — not 295 raw issues, the same way a triage console shows crash
signatures rather than every individual report.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRIAGE = Path(__file__).resolve().parent / "data" / "triage.json"

TYPE_EMOJI = {"bug": "🐞", "feature": "✨", "docs": "📝", "question": "❓", "other": "📦"}
STALE_EMOJI = {"fresh": "🟢", "aging": "🟡", "stale": "🔴", "unknown": "⚪"}


@st.cache_data
def load_triage() -> dict:
    if not TRIAGE.exists():
        return {}
    return json.loads(TRIAGE.read_text())


def main():
    st.set_page_config(page_title="L2A Issue Triage", page_icon="🧭", layout="wide")
    st.title("🧭 L2A Issue Triage")
    st.caption("Routing new contributors to the right open-source issue — "
               "clusters of distinct problems, not a wall of raw issues.")

    data = load_triage()
    if not data:
        st.warning("No data/triage.json found. Run `python -m src.pipeline` first.")
        st.stop()

    clusters = data["clusters"]
    summary = data["summary"]

    # --- top-line metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Issues", summary["issues"])
    c2.metric("Distinct problems", summary["clusters"])
    c3.metric("Beginner-friendly", summary["beginner_issues"])
    c4.metric("Needs human review", summary["needs_review"])

    # --- sidebar filters ---
    st.sidebar.header("Filters")
    types = sorted({c["classification"]["type"] for c in clusters})
    diffs = ["beginner", "intermediate", "advanced"]
    sel_types = st.sidebar.multiselect("Type", types, default=types)
    sel_diffs = st.sidebar.multiselect("Difficulty", diffs, default=diffs)
    beginner_only = st.sidebar.checkbox("Only clusters with a beginner-friendly issue")
    hide_stale = st.sidebar.checkbox("Hide stale clusters")
    query = st.sidebar.text_input("Search title").strip().lower()

    def keep(c: dict) -> bool:
        cl = c["classification"]
        if cl["type"] not in sel_types or cl["difficulty"] not in sel_diffs:
            return False
        if beginner_only and not any(i["looks_beginner"] for i in c["issues"]):
            return False
        if hide_stale and c["representative"]["staleness"] == "stale":
            return False
        if query and query not in c["representative"]["title"].lower():
            return False
        return True

    shown = [c for c in clusters if keep(c)]
    shown.sort(key=lambda c: c["size"], reverse=True)
    st.write(f"**{len(shown)}** clusters match "
             f"({sum(c['size'] for c in shown)} issues)")

    # --- cluster cards ---
    for c in shown:
        cl = c["classification"]
        rep = c["representative"]
        emoji = TYPE_EMOJI.get(cl["type"], "📦")
        review = " ⚠️ needs review" if cl["needs_review"] else ""
        header = (f"{emoji} **{cl['type']} · {cl['difficulty']}** · "
                  f"{c['size']} issue(s){review} — {rep['title']}")
        with st.expander(header):
            st.markdown(
                f"**Confidence:** {cl['confidence']} &nbsp;|&nbsp; "
                f"**Rationale:** {cl['rationale']}"
            )
            rows = []
            for i in c["issues"]:
                rows.append({
                    "": STALE_EMOJI.get(i["staleness"], "⚪"),
                    "issue": f"#{i['number']}",
                    "repo": i["source_repo"],
                    "title": i["title"],
                    "beginner": "✅" if i["looks_beginner"] else "",
                    "trace": "🧵" if i["has_error_trace"] else "",
                    "link": i["html_url"],
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={"link": st.column_config.LinkColumn("link", display_text="open")},
            )


if __name__ == "__main__":
    main()
