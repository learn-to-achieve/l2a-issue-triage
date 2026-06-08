"""
classify.py — LLM consumer: label each issue cluster.

Architectural parallel (production crash triage): label the crash *signature*
once, then apply to every instance. Here we classify the cluster
representative and fan the label out to all members — 120 clusters
means ~120 LLM calls, not 295. The clustering step pays for itself
by cutting LLM calls by more than half.

Design stance — treat the LLM as an unreliable network dependency:
  * a strict "return ONLY JSON" prompt,
  * a parser that survives code fences and chatty wrappers,
  * coercion that clamps any hallucinated label to the allowed vocab,
  * exponential backoff on rate limits + fail-fast on per-day quota,
  * a graceful low-confidence fallback so one flaky call never crashes
    the run — and low-confidence results are flagged for human review.

Two modes (CLASSIFY_MODE):
  * "single" (default): one Triage call per cluster — original behavior.
  * "multi": three composable roles behind the SAME model-agnostic backend —
      Triage   — assigns type/difficulty (the per-cluster call),
      Verifier — re-checks a label against the issue's evidence and, on
                 disagreement, flags needs_review. Runs ONLY on lower-confidence
                 labels, so the call budget stays sane (not a blind 3x).
      Router   — given a learner target (skill/difficulty), picks the
                 best-matching cluster. On-demand, not per cluster.

`staleness` is intentionally NOT asked of the model — it's a date
computation handled in normalize.py.

Usage:
    python -m src.classify                 # CLASSIFY_MODE=single (default)
    CLASSIFY_MODE=multi python -m src.classify
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

# Multi-agent mode + the band in which the Verifier runs.
CLASSIFY_MODE = os.environ.get("CLASSIFY_MODE", "single").lower()   # "single" | "multi"
VERIFY_CONF_THRESHOLD = 0.8   # multi mode: verify only labels below this confidence

# Optional distillation cost cascade: a cheap supervised model (src/prefilter.py)
# short-circuits the LLM on confident, easy cases. Off by default.
USE_PREFILTER = os.environ.get("PREFILTER", "").lower() in ("1", "true", "yes")

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

VERIFY_PROMPT = """You are double-checking a triage label on a GitHub issue.
A first pass proposed — type: {ptype}, difficulty: {pdifficulty}.
Re-read the issue and judge whether that label is correct.
Return ONLY a JSON object — no prose, no code fences — with exactly these keys:
  "agree": true or false
  "type": the correct one of {types}
  "difficulty": the correct one of {difficulties}
  "reason": one short sentence (max 20 words)

Issue title: {title}
Issue body (may be truncated):
{body}
"""

ROUTE_PROMPT = """A learner wants their first open-source issue to work on.
Target — skill/topic: {skill}; difficulty: {difficulty}.
From the candidate clusters below, pick the single BEST match.
Return ONLY a JSON object: {{"cluster_id": <int from the list>, "reason": "<one short sentence>"}}

Candidates:
{candidates}
"""


# --- defensive parsing ------------------------------------------------------

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _json_from(text: str) -> dict | None:
    """Extract the first JSON object from a possibly-fenced/chatty reply."""
    if not text:
        return None
    s = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    s = re.sub(r"```$", "", s).strip()
    m = _JSON_OBJ.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return None


def parse_response(text: str) -> dict:
    """Parse a Triage reply into a clamped label, or FALLBACK if unparseable."""
    obj = _json_from(text)
    if obj is None:
        return dict(FALLBACK)
    return coerce(obj)


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


# --- model call (shared, with backoff + daily-quota fail-fast) --------------

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


def _invoke(chat, prompt: str, log: str, max_tries: int = 6) -> str | None:
    """Call the model, returning text or None. Fail-fast on per-day quota,
    exponential backoff on per-minute limits, swallow other errors."""
    delay = 15.0
    for attempt in range(max_tries):
        try:
            resp = chat.invoke(prompt)
            return getattr(resp, "content", "") or ""
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg or "RequestsPerDay" in msg:
                print(f"  [daily-quota] exhausted — {log}")
                return None
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(f"  [rate-limit] backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_tries}) — {log}")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"  [warn] LLM error ({log}): {e}")
            return None
    return None


# --- the three roles --------------------------------------------------------

def _triage_one(chat, rec: dict, max_tries: int = 6) -> dict:
    """Triage agent: assign type/difficulty for one cluster representative."""
    prompt = PROMPT.format(
        types=TYPES, difficulties=DIFFICULTIES,
        title=rec["title"][:200], body=rec["clean_text"][:BODY_CHARS],
    )
    text = _invoke(chat, prompt, f"#{rec.get('number')}", max_tries)
    return dict(FALLBACK) if text is None else parse_response(text)


# Back-compat alias: callers (and tests) may still use _classify_one.
_classify_one = _triage_one


def _verify_one(chat, rec: dict, label: dict, max_tries: int = 4) -> dict:
    """Verifier agent: re-check a proposed label against the issue evidence.
    Returns {agree, type, difficulty, reason}. Fails open (agree) if the model
    is unavailable, so verification never *invents* a disagreement."""
    prompt = VERIFY_PROMPT.format(
        types=TYPES, difficulties=DIFFICULTIES,
        ptype=label["type"], pdifficulty=label["difficulty"],
        title=rec["title"][:200], body=rec["clean_text"][:BODY_CHARS],
    )
    text = _invoke(chat, prompt, f"verify #{rec.get('number')}", max_tries)
    if text is None:
        return {"agree": True, "type": label["type"],
                "difficulty": label["difficulty"], "reason": "verifier unavailable"}
    obj = _json_from(text) or {}
    vt = str(obj.get("type", "")).lower().strip()
    vd = str(obj.get("difficulty", "")).lower().strip()
    vt = vt if vt in TYPES else label["type"]
    vd = vd if vd in DIFFICULTIES else label["difficulty"]
    # Treat a stated "agree" that nonetheless changes a label as a disagreement.
    agree = bool(obj.get("agree", True)) and vt == label["type"] and vd == label["difficulty"]
    reason = str(obj.get("reason", "")).strip()[:160] or "(none)"
    return {"agree": agree, "type": vt, "difficulty": vd, "reason": reason}


def route(target: dict, records: list[dict], clusters: list[list[int]],
          labels: list[dict], chat=None, shortlist: int = 12) -> dict:
    """Router agent: pick the best cluster for a learner target.

    target = {"skill": <str|None>, "difficulty": <beginner|intermediate|advanced|None>}.
    Deterministically prefilters by difficulty and ranks by size, then (if a chat
    client and a skill are given) asks the model to pick from the shortlist — and
    clamps the answer to a real candidate. With no chat, returns the top
    deterministic match. Never multiplies the per-cluster budget.
    """
    tdiff = str(target.get("difficulty") or "").strip().lower()
    skill = str(target.get("skill") or "").strip()

    cands = []
    for cid, (members, label) in enumerate(zip(clusters, labels)):
        if label.get("needs_review"):
            continue
        if tdiff and label["difficulty"] != tdiff:
            continue
        cands.append({
            "cluster_id": cid, "size": len(members),
            "type": label["type"], "difficulty": label["difficulty"],
            "title": records[members[0]]["title"][:120],
        })
    cands.sort(key=lambda c: c["size"], reverse=True)
    cands = cands[:shortlist]

    if not cands:
        return {"cluster_id": None,
                "reason": f"no cluster matches difficulty={tdiff or 'any'}",
                "candidates": []}

    if not (chat and skill):
        best = cands[0]
        return {"cluster_id": best["cluster_id"],
                "reason": f"largest {best['difficulty']} cluster matching the target",
                "candidates": cands}

    listing = "\n".join(
        f'- cluster_id {c["cluster_id"]}: [{c["type"]}/{c["difficulty"]}] {c["title"]}'
        for c in cands)
    prompt = ROUTE_PROMPT.format(skill=skill or "(any)",
                                 difficulty=tdiff or "(any)", candidates=listing)
    obj = _json_from(_invoke(chat, prompt, "route", max_tries=4) or "") or {}
    valid = {c["cluster_id"] for c in cands}
    cid = obj.get("cluster_id")
    if cid not in valid:
        return {"cluster_id": cands[0]["cluster_id"],
                "reason": "router returned no valid id; fell back to top match",
                "candidates": cands}
    return {"cluster_id": cid,
            "reason": str(obj.get("reason", "")).strip()[:200] or "router pick",
            "candidates": cands}


# --- checkpointed cache -----------------------------------------------------

def _cache_file(mode: str) -> Path:
    base = MODEL_NAME.replace(":", "_").replace("/", "_")
    suffix = "_multi" if mode == "multi" else ""
    return ingest.CACHE_DIR / f"classify_{base}{suffix}.json"


# single-mode path kept for reference / back-compat
CACHE_FILE = _cache_file("single")


def _load_prefilter():
    """Lazily load the distilled pre-filter model, or None if absent/unavailable."""
    try:
        from . import prefilter
        return prefilter.load()
    except Exception as e:  # noqa: BLE001 — sklearn/joblib missing etc.
        print(f"  [prefilter] unavailable: {e}")
        return None


def _prefilter_predict(model, text: str) -> tuple[dict | None, bool]:
    from . import prefilter
    return prefilter.predict(text, model=model)


def _load_cache(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2))


def classify_clusters(records: list[dict], clusters: list[list[int]],
                      mode: str | None = None) -> list[dict]:
    """Label each cluster's representative; return one label per cluster
    (same index order as `clusters`).

    mode "single" (default) = one Triage call per cluster.
    mode "multi"            = Triage, then Verifier on lower-confidence labels.
    """
    mode = (mode or CLASSIFY_MODE).lower()
    cache_path = _cache_file(mode)
    cache = _load_cache(cache_path)
    todo = [c for c in clusters if clu._key(records[c[0]]) not in cache]
    labels: dict[str, dict] = {}
    verified = flagged = short_circuited = escalated = 0

    if todo:
        print(f"[classify] mode={mode} · prefilter={'on' if USE_PREFILTER else 'off'} · "
              f"{len(todo)} new / {len(clusters)} clusters "
              f"({len(clusters) - len(todo)} cached) via {LLM_BACKEND}:{MODEL_NAME}")
        pre = _load_prefilter() if USE_PREFILTER else None
        if USE_PREFILTER and pre is None:
            print("  [prefilter] enabled but no model found "
                  "(train via `python -m src.prefilter`); escalating all to the LLM")
        chat = None  # created lazily — an all-short-circuit run never loads the LLM
        for n, c in enumerate(todo, 1):
            rep = records[c[0]]
            label = None

            # Cost cascade: a confident cheap prediction skips the LLM entirely.
            if pre is not None:
                plabel, confident = _prefilter_predict(pre, rep.get("clean_text", ""))
                if confident:
                    label = plabel
                    short_circuited += 1

            if label is None:                                    # escalate to the LLM
                escalated += 1
                if chat is None:
                    chat = _chat()
                label = _triage_one(chat, rep)                   # role 1: Triage
                # role 2: Verifier — only for trustworthy-but-uncertain labels,
                # so we don't blindly 3x the calls. Flagged/fallback labels skip it.
                if (mode == "multi" and not label["needs_review"]
                        and label["confidence"] < VERIFY_CONF_THRESHOLD):
                    verified += 1
                    v = _verify_one(chat, rep, label)
                    if not v["agree"]:
                        flagged += 1
                        label = {**label, "needs_review": True,
                                 "rationale": (label["rationale"]
                                               + f" | verifier disagrees: {v['reason']}")[:200]}
                    time.sleep(SLEEP)

            labels[clu._key(rep)] = label
            # Persist only trusted labels; fallbacks/flagged stay uncached so a
            # rerun retries them.
            if not label["needs_review"]:
                cache[clu._key(rep)] = label
                _save_cache(cache_path, cache)
            src = label.get("source", "llm")
            print(f"  [{src}] {n}/{len(todo)}  #{rep['number']} "
                  f"-> {label['type']}/{label['difficulty']}"
                  f"{' (needs_review)' if label['needs_review'] else ''}")
            # No pacing needed after a cheap prefilter hit (no API call was made).
            if src != "prefilter" and n < len(todo):
                time.sleep(SLEEP)

        if USE_PREFILTER:
            print(f"  [prefilter] short-circuited {short_circuited}, "
                  f"escalated {escalated} to the LLM")
        if mode == "multi":
            print(f"  [verify] ran on {verified} low-confidence labels, "
                  f"flagged {flagged} disagreement(s)")
    else:
        print(f"[classify] mode={mode} · all {len(clusters)} clusters served from cache")

    return [cache.get(clu._key(records[c[0]]),
                      labels.get(clu._key(records[c[0]]), dict(FALLBACK)))
            for c in clusters]


if __name__ == "__main__":
    records, clusters = clu.cluster_issues()
    labels = classify_clusters(records, clusters)
    review = sum(1 for lab in labels if lab["needs_review"])
    from collections import Counter
    by_type = Counter(lab["type"] for lab in labels)
    print("-" * 60)
    print(f"[done] mode={CLASSIFY_MODE} · {len(labels)} clusters classified "
          f"({review} flagged for review)")
    print("by type:", dict(by_type))
