"""
cluster.py — Group near-duplicate / same-topic issues via embeddings.

Architectural parallel (Google Nest): crash fingerprinting. There,
~2,000 raw crashes collapsed into ~50 distinct signatures so an
engineer triaged groups, not noise. Here, 276 issues collapse into
N distinct problems so a newcomer sees real choices.

Design choices:
  * Greedy threshold clustering, NOT k-means — no `k` to guess, no
    random seeds, identical clusters run to run. When the output feeds
    a human triage decision, determinism beats sophistication.
  * Embeddings are cached to disk (same discipline as the raw-issue
    cache in ingest.py). Retuning SIM_THRESHOLD or re-clustering then
    costs zero API calls, and a rate-limit mid-run never loses work.
  * Embedding respects the free-tier budget: small batches, truncated
    inputs, pacing, and exponential backoff on 429.

Usage:
    python -m src.cluster
"""

import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from . import ingest
from . import normalize as norm

EMBED_MODEL = "models/gemini-embedding-001"
SIM_THRESHOLD = 0.83   # cosine; tuned for "same problem", not "same words"
BATCH = 10             # small batches keep us under the free-tier token/min burst
MAX_CHARS = 2000       # ~500 tokens/issue — plenty of signal, bounded burst
SLEEP = 8.0            # seconds between batches, to respect tokens-per-minute
CACHE_FILE = ingest.CACHE_DIR / f"embeddings_{EMBED_MODEL.split('/')[-1]}.npz"


def _key(rec: dict) -> str:
    """Stable per-issue cache key."""
    return f"{rec['source_repo']}#{rec['number']}"


# --- embedding cache --------------------------------------------------------

def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE_FILE.exists():
        return {}
    d = np.load(CACHE_FILE, allow_pickle=True)
    return {str(k): v for k, v in zip(d["keys"], d["vectors"])}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    keys = np.array(list(cache.keys()), dtype=object)
    vectors = np.array(list(cache.values()), dtype="float32")
    np.savez(CACHE_FILE, keys=keys, vectors=vectors)


# --- embedding --------------------------------------------------------------

def _embedder():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("your_"):
        raise SystemExit("GEMINI_API_KEY missing/placeholder in .env")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=key)


def _embed_batch_with_backoff(emb, texts: list[str], max_tries: int = 6) -> list[list[float]]:
    delay = 15.0
    for attempt in range(max_tries):
        try:
            return emb.embed_documents(texts)
        except Exception as e:  # noqa: BLE001 — only retry on rate-limit
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"  [rate-limit] backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_tries})")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise SystemExit("Still rate-limited after retries — rerun later; "
                     "cached batches are saved so it resumes.")


def embed_records(records: list[dict]) -> np.ndarray:
    """Embed records (cached, incremental); return L2-normalized matrix
    aligned to `records` order."""
    cache = _load_cache()
    todo = [r for r in records if _key(r) not in cache]
    if todo:
        print(f"[embed] {len(todo)} new / {len(records)} total "
              f"({len(records) - len(todo)} cached)")
        emb = _embedder()
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            texts = [r["clean_text"][:MAX_CHARS] for r in chunk]
            vecs = _embed_batch_with_backoff(emb, texts)
            for r, v in zip(chunk, vecs):
                cache[_key(r)] = np.asarray(v, dtype="float32")
            _save_cache(cache)  # checkpoint — quota spent is never lost
            print(f"  [embed] {min(i + BATCH, len(todo))}/{len(todo)} new")
            if i + BATCH < len(todo):
                time.sleep(SLEEP)
    else:
        print(f"[embed] all {len(records)} issues served from cache")

    arr = np.array([cache[_key(r)] for r in records], dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --- clustering -------------------------------------------------------------

def greedy_cluster(vecs: np.ndarray, threshold: float = SIM_THRESHOLD) -> list[list[int]]:
    """Deterministic greedy clustering over normalized vectors via FAISS.

    Walk points in index order. Each unassigned point seeds a new
    cluster and absorbs every still-unassigned point whose cosine
    similarity clears `threshold`. Fixed order → stable output.
    """
    import faiss

    n, d = vecs.shape
    index = faiss.IndexFlatIP(d)   # inner product == cosine (vecs normalized)
    index.add(vecs)

    assigned = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        lims, _D, ids = index.range_search(vecs[i:i + 1], threshold)
        members = [int(j) for j in ids[lims[0]:lims[1]] if not assigned[j]]
        if i not in members:
            members.append(i)
        for j in members:
            assigned[j] = True
        clusters.append(sorted(members))
    return clusters


def cluster_issues(records: list[dict] | None = None):
    records = records if records is not None else norm.normalize()
    print(f"[cluster] embedding {len(records)} issues via {EMBED_MODEL} ...")
    vecs = embed_records(records)
    clusters = greedy_cluster(vecs)
    clusters.sort(key=len, reverse=True)
    return records, clusters


if __name__ == "__main__":
    records, clusters = cluster_issues()
    singletons = sum(1 for c in clusters if len(c) == 1)
    print("-" * 60)
    print(f"[done] {len(records)} issues -> {len(clusters)} clusters "
          f"({singletons} singletons)")
    print("Top 5 clusters:")
    for rank, c in enumerate(clusters[:5], 1):
        rep = records[c[0]]
        print(f"  {rank}. size={len(c):>3}  #{rep['number']} [{rep['source_repo']}]")
        print(f"      {rep['title'][:80]}")
