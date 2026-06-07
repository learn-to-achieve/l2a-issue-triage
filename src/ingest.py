"""
ingest.py — Multi-source GitHub issue ingestion.

Architectural parallel (Google Nest): this layer plays the role of
multi-source crash ingestion (Coverity / Plx Workflows / GoCrash).
Instead of crash reports from three systems, we pull open issues
from multiple GitHub repos. Everything downstream assumes this
layer produces raw, uniform JSON.

Usage:
    python -m src.ingest                # uses DEFAULT_REPOS
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


def _headers() -> dict:
    """Auth is optional: 60 req/hr without a token, 5000 with one."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _cache_path(repo: str) -> Path:
    return CACHE_DIR / f"{repo.replace('/', '__')}.json"


def fetch_issues(repo: str, force_refresh: bool = False) -> list[dict]:
    """Fetch open issues for one repo, with local caching.

    Caching matters for two reasons:
    1. GitHub rate limits (the demo must not die on stage).
    2. Reproducibility — same input data every run while iterating
       on clustering/classification downstream.
    """
    cache = _cache_path(repo)
    if cache.exists() and not force_refresh:
        print(f"[cache] {repo} <- {cache.name}")
        return json.loads(cache.read_text())

    issues: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params = {"state": "open", "per_page": PER_PAGE, "page": page}
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


def ingest(repos: list[str] | None = None, force_refresh: bool = False) -> list[dict]:
    """Pull issues across all source repos into one raw collection.

    Each issue is tagged with its source repo — downstream layers
    must never lose track of provenance (same rule as keeping
    device/build metadata attached to crash signatures).
    """
    repos = repos or DEFAULT_REPOS
    all_issues: list[dict] = []
    for repo in repos:
        for issue in fetch_issues(repo, force_refresh=force_refresh):
            issue["_source_repo"] = repo
            all_issues.append(issue)
    print(f"[done] {len(all_issues)} issues from {len(repos)} repos")
    return all_issues


if __name__ == "__main__":
    target_repos = sys.argv[1:] or None
    ingest(target_repos)
