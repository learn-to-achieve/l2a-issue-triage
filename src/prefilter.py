"""
prefilter.py — supervised pre-filter (a distillation cost cascade).

Trains a small, cheap model (TF-IDF + LogisticRegression) on the LLM's OWN
labels already in data/triage.json, then uses it to short-circuit the LLM at
classify time: if the cheap model is confident enough, use its prediction and
SKIP the LLM call; otherwise escalate to the LLM.

IMPORTANT: this model IMITATES the LLM to save cost — it does NOT outperform it.
The LLM's labels are the training target (the teacher); the pre-filter is a
distilled student that's right often enough on the easy cases to be worth
skipping a paid/slow call. The LLM remains the quality ceiling and the
escalation path for anything uncertain.

Train (writes models/prefilter.joblib, gitignored):
    python -m src.prefilter
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "prefilter.joblib"
TRIAGE = ROOT / "data" / "triage.json"

# Confidence each head must clear to short-circuit the LLM.
THRESHOLD = float(os.environ.get("PREFILTER_THRESHOLD", "0.7"))

_LOADED = None  # process-level cache of the loaded bundle


def _training_data() -> tuple[list[str], list[str], list[str]]:
    """Join normalized issue text to the LLM labels in triage.json.

    Each member issue inherits its cluster's (type, difficulty); features are
    the normalized text. Requires data/triage.json + the ingest cache locally.
    """
    from . import normalize as norm
    triage = json.loads(TRIAGE.read_text())
    label_by_key: dict[str, tuple[str, str]] = {}
    for c in triage["clusters"]:
        cl = c["classification"]
        for iss in c["issues"]:
            label_by_key[f'{iss["source_repo"]}#{iss["number"]}'] = (cl["type"], cl["difficulty"])

    X, y_type, y_diff = [], [], []
    for r in norm.normalize():
        key = f'{r["source_repo"]}#{r["number"]}'
        if key in label_by_key:
            X.append(r["clean_text"])
            y_type.append(label_by_key[key][0])
            y_diff.append(label_by_key[key][1])
    return X, y_type, y_diff


def build(X: list[str], y_type: list[str], y_diff: list[str]) -> dict:
    """Train two TF-IDF + LogisticRegression heads (type, difficulty)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def head():
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    return {
        "type": head().fit(X, y_type),
        "difficulty": head().fit(X, y_diff),
        "n": len(X),
    }


def train(out: Path = MODEL_PATH) -> int:
    import joblib
    X, y_type, y_diff = _training_data()
    if len(X) < 10 or len(set(y_type)) < 2:
        raise SystemExit(f"not enough labeled data to train (got {len(X)} samples, "
                         f"{len(set(y_type))} type classes)")
    bundle = build(X, y_type, y_diff)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return bundle["n"]


def load():
    """Load the trained bundle, or None if it hasn't been trained yet."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not MODEL_PATH.exists():
        return None
    import joblib
    _LOADED = joblib.load(MODEL_PATH)
    return _LOADED


def predict(text: str, model=None) -> tuple[dict | None, bool]:
    """Return (label, confident). label is None if there's no model.

    `confident` is True only when BOTH heads clear THRESHOLD — only then is it
    safe to skip the LLM. The label mirrors the LLM label shape (+ source tag).
    """
    m = model if model is not None else load()
    if not m:
        return None, False
    tm, dm = m["type"], m["difficulty"]
    pt = tm.predict_proba([text])[0]
    pd = dm.predict_proba([text])[0]
    ti, di = pt.argmax(), pd.argmax()
    t, d = tm.classes_[ti], dm.classes_[di]
    ct, cd = float(pt[ti]), float(pd[di])
    confident = ct >= THRESHOLD and cd >= THRESHOLD
    label = {
        "type": str(t),
        "difficulty": str(d),
        "confidence": round(min(ct, cd), 2),
        "rationale": f"pre-filter (distilled) prediction; type p={ct:.2f}, diff p={cd:.2f}",
        "needs_review": not confident,
        "source": "prefilter",
    }
    return label, confident


if __name__ == "__main__":
    n = train()
    print(f"[prefilter] trained on {n} samples -> {MODEL_PATH.relative_to(ROOT)}")
