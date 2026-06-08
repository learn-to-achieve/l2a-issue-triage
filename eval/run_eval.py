"""
eval/run_eval.py — score the triage classifier against a HUMAN golden set.

Ground truth lives in eval/golden.jsonl and is YOUR hand judgment — never
LLM-generated, never copied from data/triage.json. This script joins the golden
labels to the model's predictions (data/triage.json) on cluster_id and reports
per-type and per-difficulty precision / recall / F1 plus confusion matrices.

If the golden file is still the template (no filled rows), it reports that and
exits cleanly — so it's safe to run before you've labeled anything.

Run (from repo root):
    python -m eval.run_eval
    # or: python eval/run_eval.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden.jsonl"
TRIAGE = ROOT / "data" / "triage.json"

TYPES = ["bug", "feature", "docs", "question", "other"]
DIFFS = ["beginner", "intermediate", "advanced"]


def load_golden() -> list[dict]:
    """Read golden.jsonl, skipping # comments and blank lines."""
    rows: list[dict] = []
    if not GOLDEN.exists():
        return rows
    for line in GOLDEN.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            print(f"[warn] skipping unparseable line: {s[:60]}... ({e})")
    return rows


def filled(rows: list[dict]) -> list[dict]:
    """Keep only rows whose human labels are valid (TODO/empty are 'unfilled')."""
    out = []
    for r in rows:
        t = str(r.get("human_type", "")).lower()
        d = str(r.get("human_difficulty", "")).lower()
        if t in TYPES and d in DIFFS:
            out.append({**r, "human_type": t, "human_difficulty": d})
    return out


def load_predictions() -> dict[int, dict]:
    data = json.loads(TRIAGE.read_text())
    return {c["cluster_id"]: c["classification"] for c in data["clusters"]}


def _report(dim: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n================ {dim} ================")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion matrix (rows = human truth, cols = model):")
    print("  labels:", labels)
    for lab, row in zip(labels, cm):
        print(f"  {lab:>12} {row.tolist()}")


def main() -> None:
    golden = load_golden()
    gl = filled(golden)
    print(f"golden rows: {len(golden)} present, {len(gl)} filled (human-labeled)")

    if not gl:
        print("\nGolden set is still a TEMPLATE — no human labels yet.")
        print("Fill eval/golden.jsonl to ~30 rows with your own judgment, then re-run.")
        return

    if not TRIAGE.exists():
        print(f"\n[error] {TRIAGE} not found — run `python -m src.pipeline` first.")
        return
    preds = load_predictions()

    yt_type, yp_type, yt_diff, yp_diff, missing = [], [], [], [], []
    for r in gl:
        cid = r["cluster_id"]
        if cid not in preds:
            missing.append(cid)
            continue
        yt_type.append(r["human_type"]);   yp_type.append(preds[cid]["type"])
        yt_diff.append(r["human_difficulty"]); yp_diff.append(preds[cid]["difficulty"])

    if missing:
        print(f"[warn] {len(missing)} golden cluster_id(s) not in triage.json: {missing}")
    if not yt_type:
        print("[error] no golden rows matched a cluster in triage.json — nothing to score.")
        return

    print(f"evaluating {len(yt_type)} matched clusters")
    _report("TYPE", yt_type, yp_type, TYPES)
    _report("DIFFICULTY", yt_diff, yp_diff, DIFFS)


if __name__ == "__main__":
    main()
