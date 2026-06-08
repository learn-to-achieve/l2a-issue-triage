"""
normalize.py — Clean raw GitHub issues into ML-ready records.

Architectural parallel (Google Nest): this layer flattens nested,
multi-source telemetry into uniform rows the way crash signatures
were normalized before clustering. Raw issue JSON is noisy —
HTML comment templates, fenced code blocks, checklist boilerplate.
Similarity downstream must be driven by the human description of
the problem, not by the repo's issue-template scaffolding.

Two cheap, pre-computed signals (`looks_beginner`, `has_error_trace`)
ride along on each record. They give the LLM classifier a head start
and let the triage board filter without a model call.

Usage:
    python -m src.normalize        # prints usable-record count + samples
"""

import re
from datetime import datetime, timezone
from typing import Any

from . import ingest

# --- cleaning ---------------------------------------------------------------

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)        # issue-template hints
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)            # tracebacks / snippets
_HTML_TAG = re.compile(r"<[^>]+>")                           # stray markup
_WS = re.compile(r"\s+")


def clean_body(body: str | None) -> str:
    """Strip template noise so embeddings reflect the human description.

    Order matters: code fences are removed here, but `has_error_trace`
    is computed on the RAW body first (see normalize_one) — otherwise
    we'd delete the very traceback we want to detect.
    """
    text = body or ""
    text = _HTML_COMMENT.sub(" ", text)
    text = _CODE_FENCE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return _WS.sub(" ", text).strip()


# --- signals ----------------------------------------------------------------

# Conservative: only labels that explicitly mark entry-level work.
_BEGINNER_HINTS = (
    "good first issue",
    "good-first-issue",
    "good first",
    "beginner",
    "starter",
    "easy",
    "help wanted",
)

# Patterns that signal a real stack trace / exception in the body.
_TRACE_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'\bFile ".*", line \d+'),     # python frames
    re.compile(r"\b[A-Z]\w*(Error|Exception)\b"),   # FooError / BarException
    re.compile(r"^\s+at .+\(.*:\d+\)", re.MULTILINE),  # JS/Java frames
]


def label_names(issue: dict[str, Any]) -> list[str]:
    """GitHub labels are objects; flatten to lowercase names."""
    out = []
    for lab in issue.get("labels") or []:
        name = lab.get("name") if isinstance(lab, dict) else str(lab)
        if name:
            out.append(name.lower())
    return out


def looks_beginner(labels: list[str]) -> bool:
    return any(hint in lab for lab in labels for hint in _BEGINNER_HINTS)


def has_error_trace(raw_body: str | None) -> bool:
    text = raw_body or ""
    return any(p.search(text) for p in _TRACE_PATTERNS)


def staleness(updated_at: str | None, now: datetime | None = None) -> str:
    """Bucket an issue by time since last activity — a date computation,
    deliberately NOT an LLM judgment. fresh <30d, aging <180d, else stale."""
    if not updated_at:
        return "unknown"
    now = now or datetime.now(timezone.utc)
    try:
        dt = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return "unknown"
    days = (now - dt).days
    if days < 30:
        return "fresh"
    if days < 180:
        return "aging"
    return "stale"


# --- normalization ----------------------------------------------------------

MIN_TEXT_LEN = 20  # drop near-empty issues that can't be embedded meaningfully


def normalize_one(issue: dict[str, Any]) -> dict[str, Any]:
    raw_body = issue.get("body")
    labels = label_names(issue)
    title = (issue.get("title") or "").strip()
    body = clean_body(raw_body)
    # Embedding text leads with the title (highest signal), then body.
    clean_text = _WS.sub(" ", f"{title}. {body}").strip()

    return {
        "source_repo": issue.get("_source_repo"),
        "number": issue.get("number"),
        "title": title,
        "html_url": issue.get("html_url"),
        "labels": labels,
        "comments": issue.get("comments", 0),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "clean_text": clean_text,
        "looks_beginner": looks_beginner(labels),
        "has_error_trace": has_error_trace(raw_body),
        "staleness": staleness(issue.get("updated_at")),
    }


def normalize(records: list[dict] | None = None) -> list[dict[str, Any]]:
    """Return usable, normalized records (drops near-empty bodies)."""
    raw = records if records is not None else ingest.ingest()
    out = []
    for issue in raw:
        if "pull_request" in issue:  # defensive: PRs shouldn't reach here
            continue
        rec = normalize_one(issue)
        if len(rec["clean_text"]) >= MIN_TEXT_LEN:
            out.append(rec)
    return out


if __name__ == "__main__":
    recs = normalize()
    beginner = sum(r["looks_beginner"] for r in recs)
    traces = sum(r["has_error_trace"] for r in recs)
    print(f"[normalize] {len(recs)} usable records "
          f"({beginner} beginner-friendly, {traces} with error traces)")
    print("-" * 60)
    for r in recs[:3]:
        print(f"#{r['number']} [{r['source_repo']}] beginner={r['looks_beginner']} "
              f"trace={r['has_error_trace']}")
        print(f"  title: {r['title'][:80]}")
        print(f"  text : {r['clean_text'][:120]}...")
        print()
