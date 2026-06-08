"""
classify.py — LLM consumer: label each issue cluster.

Architectural parallel (production crash triage): label the crash *signature*
once, then apply to every instance. Here we classify the cluster
representative and fan the label out to all members — 115 clusters
means 115 Gemini calls, not 275. The clustering step pays for itself
by cutting LLM calls by more than half.

Design stance — treat the LLM as an unreliable network dependency:
  * a strict "return ONLY JSON" prompt,
  * a parser that survives code fences and chatty wrappers,
  * coercion that clamps any hallucinated label to the allowed vocab,
  * exponential backoff on rate limits,
  * a graceful low-confidence fallback so one flaky call never crashes
    the run — and low-confidence results are flagged for human review
    rather than trusted blindly.

`staleness` is intentionally NOT asked of the model — it's a date
computation handled in normalize.py.

Usage:
    python -m src.classify        # classifies clusters built by cluster.py
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from . import cluster as clu
from . import ingest

# Model-agnostic backend switch. The cloud classifier hit a per-day quota wall
# mid-run; because this layer is a clean interface (defensive parser, output
# clamping, cluster-once pattern), only the *client* swaps — same prompt, same
# parser, zero downstream changes. LLM_BACKEND=ollama runs a local model on the
# GPU (no per-call cost, full data control); default stays cloud.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini").lower()   # "gemini" | "ollama"
GEMINI_MODEL = "gemini-2.5-flash-lite"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")

MODEL_NAME = OLLAMA_MODEL if LLM_BACKEND == "ollama" else GEMINI_MODEL
# Cloud needs pacing for tokens/min; local has no TPM ceiling — rip through it.
SLEEP = 0.0 if LLM_BACKEND == "ollama" else 5.0
BODY_CHARS = 1500      # truncate body sent to the model
CACHE_FILE = ingest.CACHE_DIR / f"classify_{MODEL_NAME.replace(':', '_').replace('/', '_')}.json"

TYPES = ["bug", "feature", "docs", "question", "other"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]

FALLBACK = {
    "type": "other",
    "difficulty": "intermediate",
    "confidence": 0.0,
    "rationale": "classification failed; flagged for human review",
    "needs_review": True,
}

PROMPT = """You are triaging an open-source GitHub issue for a new contributor.
Return ONLY a JSON object — no prose, no markdown, no code fences — with exactly these keys:
  "type": one of {types}
  "difficulty": one of {difficulties}
  "confidence": a number from 0 to 1
  "rationale": one short sentence (max 20 words)

Issue title: {title}
Issue body (may be truncated):
{body}
"""


# --- defensive parsing ------------------------------------------------------

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict:
    """Extract a JSON object from a possibly-chatty / fenced LLM reply.

    Survives ```json fences and leading/trailing prose. Returns the
    FALLBACK label if nothing parseable is found.
    """
    if not text:
        return dict(FALLBACK)
    cleaned = text.strip()
    # Drop code fences if present.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    m = _JSON_OBJ.search(cleaned)
    if not m:
        return dict(FALLBACK)
    try:
        return coerce(json.loads(m.group(0)))
    except (json.JSONDecodeError, TypeError):
        return dict(FALLBACK)


def coerce(obj: dict) -> dict:
    """Clamp model output to the allowed vocabulary and ranges.

    A hallucinated type/difficulty is forced into vocab; an out-of-range
    or non-numeric confidence becomes 0. Low confidence flags review.
    """
    t = str(obj.get("type", "")).lower().strip()
    d = str(obj.get("difficulty", "")).lower().strip()
    t = t if t in TYPES else "other"
    d = d if d in DIFFICULTIES else "intermediate"
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    rationale = str(obj.get("rationale", "")).strip()[:200] or "(none)"
    return {
        "type": t,
        "difficulty": d,
        "confidence": round(conf, 2),
        "rationale": rationale,
        "needs_review": conf < 0.5,
    }


# --- model call -------------------------------------------------------------

def _chat():
    """Return a LangChain chat client. Both backends expose the same
    .invoke(prompt) -> .content surface, so nothing downstream changes."""
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("your_"):
        raise SystemExit("GEMINI_API_KEY missing/placeholder in .env")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=key, temperature=0)


def _classify_one(chat, rec: dict, max_tries: int = 6) -> dict:
    prompt = PROMPT.format(
        types=TYPES, difficulties=DIFFICULTIES,
        title=rec["title"][:200],
        body=rec["clean_text"][:BODY_CHARS],
    )
    delay = 15.0
    for attempt in range(max_tries):
        try:
            resp = chat.invoke(prompt)
            return parse_response(getattr(resp, "content", "") or "")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # Per-DAY quota won't recover by waiting minutes — fail fast so the
            # run finishes in seconds with fallbacks instead of grinding hours.
            if "PerDay" in msg or "RequestsPerDay" in msg:
                print(f"  [daily-quota] exhausted — fallback for #{rec['number']}")
                return dict(FALLBACK)
            # Per-MINUTE rate limit: back off and retry.
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(f"  [rate-limit] backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_tries})")
                time.sleep(delay)
                delay *= 2
                continue
            # Any other error: don't crash the whole run on one bad issue.
            print(f"  [warn] classify error on #{rec['number']}: {e}")
            return dict(FALLBACK)
    return dict(FALLBACK)


# --- checkpointed cache -----------------------------------------------------

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def classify_clusters(records: list[dict], clusters: list[list[int]]) -> list[dict]:
    """Classify each cluster's representative; return one label per cluster
    (same index order as `clusters`)."""
    cache = _load_cache()
    todo = [c for c in clusters if clu._key(records[c[0]]) not in cache]
    labels: dict[str, dict] = {}
    if todo:
        print(f"[classify] {len(todo)} new / {len(clusters)} clusters "
              f"({len(clusters) - len(todo)} cached) via {LLM_BACKEND}:{MODEL_NAME}")
        chat = _chat()
        for n, c in enumerate(todo, 1):
            rep = records[c[0]]
            label = _classify_one(chat, rep)
            labels[clu._key(rep)] = label
            # Only PERSIST real labels — fallbacks (quota/parse failures) stay
            # uncached so a later rerun actually retries them.
            if not label["needs_review"]:
                cache[clu._key(rep)] = label
                _save_cache(cache)
            print(f"  [classify] {n}/{len(todo)}  #{rep['number']} "
                  f"-> {label['type']}/{label['difficulty']}"
                  f"{' (fallback)' if label['needs_review'] else ''}")
            if n < len(todo):
                time.sleep(SLEEP)
    else:
        print(f"[classify] all {len(clusters)} clusters served from cache")

    # Merge persisted cache with this run's in-memory (incl. fallback) labels.
    return [cache.get(clu._key(records[c[0]]), labels.get(clu._key(records[c[0]]), dict(FALLBACK)))
            for c in clusters]


if __name__ == "__main__":
    records, clusters = clu.cluster_issues()
    labels = classify_clusters(records, clusters)
    review = sum(1 for lab in labels if lab["needs_review"])
    from collections import Counter
    by_type = Counter(lab["type"] for lab in labels)
    print("-" * 60)
    print(f"[done] {len(labels)} clusters classified "
          f"({review} flagged for review)")
    print("by type:", dict(by_type))
