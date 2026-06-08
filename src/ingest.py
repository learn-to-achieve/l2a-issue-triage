"""
ingest.py — Multi-source GitHub issue ingestion.

Architectural parallel (production crash triage): this layer plays the role
of multi-source crash ingestion from several internal reporting systems.
Instead of crash reports from multiple systems, we pull open issues
from multiple GitHub repos. Everything downstream assumes this
layer produces raw, uniform JSON.

Two passes per repo (deduped by issue number): the newest open issues,
plus a dedicated `good first issue` pass so the beginner signal isn't
empty just because the most-recent issues happen to be unlabeled.

Usage:
    python -m src.ingest                # DEFAULT_REPOS, cached if present
    python -m src.ingest --force        # re-pull, ignoring cache
    python -m src.ingest owner/repo ... # explicit repos
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
GITHUB_API = "https://api.github.com"

# Beginner-friendly, high-traffic repos with healthy issue volume.
DEFAULT_REPOS = [
    "streamlit/streamlit",
    "langchain-ai/langchain",
    "pandas-dev/pandas",
]

PER_PAGE = 100
MAX_PAGES = 2  # ~200 issues per repo is plenty for the demo

# Entry-level labels for the beginner pass. "good first issue" is the
# canonical one, but it's often empty; "help wanted" is the practical
# fallback these repos actually use. (Verified via the search API:
# good-first-issue was ~0 open across all three; langchain uses help-wanted.)
# Each is a separate query — GitHub's `labels` filter is AND, not OR.
BEGINNER_LABELS = ["good first issue", "help wanted"]


def _headers() -> dict:
    """Auth is optional: 60 req/hr without a token, 5000 with one."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _cache_path(repo: str, labels: str | None = None) -> Path:
    """Label-filtered passes get their own cache file so the two passes
    never overwrite each other."""
    base = repo.replace("/", "__")
    if labels:
        slug = labels.replace(" ", "-").replace("/", "-")
        base = f"{base}__label_{slug}"
    return CACHE_DIR / f"{base}.json"


def fetch_issues(repo: str, force_refresh: bool = False,
                 labels: str | None = None) -> list[dict]:
    """Fetch open issues for one repo, with local caching.

    Caching matters for two reasons:
    1. GitHub rate limits (the demo must not die on stage).
    2. Reproducibility — same input data every run while iterating
       on clustering/classification downstream.

    `labels` (e.g. "good first issue") restricts the pull to issues
    carrying that label and is cached separately.
    """
    cache = _cache_path(repo, labels)
    if cache.exists() and not force_refresh:
        print(f"[cache] {repo} <- {cache.name}")
        return json.loads(cache.read_text())

    issues: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params = {"state": "open", "per_page": PER_PAGE, "page": page}
        if labels:
            params["labels"] = labels
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)

        if resp.status_code == 403:
            # Rate limited — fail soft, return what we have.
            print(f"[warn] rate limited on {repo}; got {len(issues)} issues")
            break
        resp.raise_for_status()

        batch = resp.json()
        if not batch:
            break

        # The issues endpoint also returns PRs; a PR has a
        # 'pull_request' key. Filter them out — same idea as
        # filtering non-crash noise out of raw telemetry.
        issues.extend(i for i in batch if "pull_request" not in i)
        time.sleep(0.5)  # be polite

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(issues, indent=2))
    print(f"[fetch] {repo}: {len(issues)} issues -> {cache.name}")
    return issues


def _merge_by_number(*issue_lists: list[dict]) -> list[dict]:
    """Dedupe issues by number, keeping the first occurrence.

    The newest pass and the beginner-label pass overlap; an issue must
    appear once, with its labels intact, regardless of which pass found
    it first.
    """
    seen: dict[int, dict] = {}
    for lst in issue_lists:
        for issue in lst:
            num = issue.get("number")
            if num is not None and num not in seen:
                seen[num] = issue
    return list(seen.values())


def ingest(repos: list[str] | None = None, force_refresh: bool = False) -> list[dict]:
    """Pull issues across all source repos into one raw collection.

    Two passes per repo — newest open issues plus a `good first issue`
    pass — deduped by number. Each issue is tagged with its source repo:
    downstream layers must never lose track of provenance (same rule as
    keeping device/build metadata attached to crash signatures).
    """
    repos = repos or DEFAULT_REPOS
    all_issues: list[dict] = []
    for repo in repos:
        newest = fetch_issues(repo, force_refresh=force_refresh)
        beginner_passes = [
            fetch_issues(repo, force_refresh=force_refresh, labels=lbl)
            for lbl in BEGINNER_LABELS
        ]
        beginner_n = sum(len(p) for p in beginner_passes)
        merged = _merge_by_number(newest, *beginner_passes)
        for issue in merged:
            issue["_source_repo"] = repo
            all_issues.append(issue)
        print(f"[repo] {repo}: {len(newest)} newest + {beginner_n} beginner-labeled "
              f"-> {len(merged)} unique")
    print(f"[done] {len(all_issues)} issues from {len(repos)} repos")
    return all_issues


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    target_repos = [a for a in args if not a.startswith("--")] or None
    ingest(target_repos, force_refresh=force)
