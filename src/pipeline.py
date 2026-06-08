"""
pipeline.py — Chain ingest -> normalize -> cluster -> classify.

Produces data/triage.json: the cluster-level triage board the
Streamlit app reads. One record per cluster, with its Gemini label
and the member issues (each carrying the cheap pre-computed signals).

Usage:
    python -m src.pipeline
"""

import json
from pathlib import Path

from . import classify, cluster, normalize

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "triage.json"


def _issue_view(rec: dict) -> dict:
    """The per-issue fields the board needs (drops the bulky clean_text)."""
    return {
        "number": rec["number"],
        "title": rec["title"],
        "html_url": rec["html_url"],
        "source_repo": rec["source_repo"],
        "labels": rec["labels"],
        "looks_beginner": rec["looks_beginner"],
        "has_error_trace": rec["has_error_trace"],
        "staleness": rec["staleness"],
    }


def run() -> dict:
    records = normalize.normalize()                 # ingest happens inside
    records, clusters = cluster.cluster_issues(records)
    labels = classify.classify_clusters(records, clusters)

    out_clusters = []
    for cid, (members, label) in enumerate(zip(clusters, labels)):
        rep = records[members[0]]
        out_clusters.append({
            "cluster_id": cid,
            "size": len(members),
            "classification": label,
            "representative": _issue_view(rep),
            "issues": [_issue_view(records[i]) for i in members],
        })

    result = {
        "summary": {
            "issues": len(records),
            "clusters": len(clusters),
            "beginner_issues": sum(r["looks_beginner"] for r in records),
            "needs_review": sum(1 for lab in labels if lab["needs_review"]),
        },
        "clusters": out_clusters,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    s = result["summary"]
    print("-" * 60)
    print(f"[pipeline] {s['issues']} issues -> {s['clusters']} clusters "
          f"({s['beginner_issues']} beginner, {s['needs_review']} need review)")
    print(f"[pipeline] wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")
    print("Top 5 clusters by size:")
    for c in sorted(result["clusters"], key=lambda x: x["size"], reverse=True)[:5]:
        cl = c["classification"]
        print(f"  size={c['size']:>3}  {cl['type']}/{cl['difficulty']} "
              f"(conf {cl['confidence']})  {c['representative']['title'][:60]}")
