"""
api/main.py — read-only FastAPI over data/triage.json.

A JSON API surface for the same triage output the Streamlit board reads, fully
decoupled from Streamlit (different process, different deps). Useful for
integrations, scripts, or a different frontend.

Run:
    pip install -r api/requirements-api.txt
    uvicorn api.main:app --reload --port 8000
    # interactive docs at http://localhost:8000/docs
"""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
TRIAGE = ROOT / "data" / "triage.json"


# --- response models --------------------------------------------------------

class Classification(BaseModel):
    type: str
    difficulty: str
    confidence: float
    rationale: str
    needs_review: bool
    source: str | None = None


class Issue(BaseModel):
    number: int
    title: str
    html_url: str | None = None
    source_repo: str
    labels: list[str] = []
    looks_beginner: bool
    has_error_trace: bool
    staleness: str


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    classification: Classification
    representative: Issue


class Cluster(ClusterSummary):
    issues: list[Issue]


class Summary(BaseModel):
    issues: int
    clusters: int
    beginner_issues: int
    needs_review: int


class Health(BaseModel):
    status: str
    data_present: bool


# --- data access ------------------------------------------------------------

@lru_cache(maxsize=1)
def _data() -> dict | None:
    if not TRIAGE.exists():
        return None
    return json.loads(TRIAGE.read_text())


def _require() -> dict:
    d = _data()
    if d is None:
        raise HTTPException(status_code=503,
                            detail="data/triage.json not found — run `python -m src.pipeline`")
    return d


app = FastAPI(title="L2A Issue Triage API", version="1.0",
              description="Read-only API over the triage pipeline output.")


# --- endpoints --------------------------------------------------------------

@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", data_present=TRIAGE.exists())


@app.get("/summary", response_model=Summary)
def summary() -> Summary:
    return Summary(**_require()["summary"])


@app.get("/clusters", response_model=list[ClusterSummary])
def clusters(
    type: str | None = Query(None, description="filter by classification type"),
    difficulty: str | None = Query(None, description="filter by difficulty"),
    beginner: bool | None = Query(None, description="clusters with a beginner-friendly issue"),
    stale: bool | None = Query(None, description="filter on representative staleness == stale"),
    q: str | None = Query(None, description="substring match on representative title"),
):
    data = _require()
    out = []
    for c in data["clusters"]:
        cl = c["classification"]
        if type and cl["type"] != type:
            continue
        if difficulty and cl["difficulty"] != difficulty:
            continue
        if beginner is not None and beginner != any(i["looks_beginner"] for i in c["issues"]):
            continue
        if stale is not None and stale != (c["representative"]["staleness"] == "stale"):
            continue
        if q and q.lower() not in c["representative"]["title"].lower():
            continue
        out.append(ClusterSummary(cluster_id=c["cluster_id"], size=c["size"],
                                  classification=cl, representative=c["representative"]))
    return out


@app.get("/clusters/{cluster_id}", response_model=Cluster)
def cluster(cluster_id: int) -> Cluster:
    for c in _require()["clusters"]:
        if c["cluster_id"] == cluster_id:
            return Cluster(**c)
    raise HTTPException(status_code=404, detail=f"cluster {cluster_id} not found")


@app.get("/issues", response_model=list[Issue])
def issues(
    beginner: bool | None = Query(None, description="only beginner-friendly issues"),
    q: str | None = Query(None, description="substring match on title"),
):
    out = []
    for c in _require()["clusters"]:
        for i in c["issues"]:
            if beginner is not None and beginner != i["looks_beginner"]:
                continue
            if q and q.lower() not in i["title"].lower():
                continue
            out.append(i)
    return out
