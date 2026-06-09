# L2A Issue Triage

AI-assisted triage agent that routes new open-source contributors to the right GitHub issue.

**Live demo:** https://l2a-issue-triage-lcpy7y5kkq-uw.a.run.app/

Built at [Learn to Achieve](https://l2a.me), a 501(c)(3) helping learners make real contributions from day one.

## Why

A new contributor faces hundreds of open issues: stale ones, duplicates, ones far beyond their level. They bounce around and contribute nothing. This is a triage problem — the same one production engineering teams face with crash reports.

## Architecture

```
GitHub API (multi-repo) --> ingest.py     (raw collection, cached)
                        --> normalize.py  (clean, extract signals)
                        --> cluster.py    (embeddings + FAISS unsupervised clustering)
                        --> classify.py   (Gemini: type/difficulty/staleness)
                        --> app.py        (Streamlit triage board)
```

The pattern mirrors a production crash-triage system:
multi-source ingest -> normalization -> fingerprint clustering -> LLM consumer.
There it cut triage from ~48 hours to under 4. Here it routes learners to
their first contribution.

## Run

```
pip install -r requirements.txt
cp .env.example .env   # add your Gemini API key
python -m src.ingest
python -m src.pipeline
streamlit run app.py
```

## Evaluation

The classifier is measured against a **human golden set** — `eval/golden.jsonl`
— not against itself. The ground-truth labels are filled in **by hand** after
reading each cluster's issues; they are **never LLM-generated and never copied
from `data/triage.json`**. If the golden labels came from the model, the
evaluation would be circular and meaningless — the golden set exists precisely
to be an independent human judgment to measure the model against.

```
# 1. Fill eval/golden.jsonl to ~30 rows by hand (the committed file is a template).
# 2. Score the model's predictions against your labels:
python -m eval.run_eval
```

`run_eval.py` joins golden labels to predictions on `cluster_id` and prints
per-type and per-difficulty precision / recall / F1 and confusion matrices
(via `sklearn.metrics`). Rows still marked `TODO` are treated as unfilled and
skipped, so you can label incrementally; with zero filled rows it just reports
that the golden set is still a template.

## API (FastAPI)

A read-only JSON API over the same `data/triage.json`, decoupled from Streamlit
(separate process and dependencies).

```
pip install -r api/requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

Endpoints (interactive docs at `http://localhost:8000/docs`):

| Endpoint | Notes |
|----------|-------|
| `GET /health` | liveness + whether triage data is present |
| `GET /summary` | issue/cluster/beginner/needs-review counts |
| `GET /clusters` | filters: `type`, `difficulty`, `beginner`, `stale`, `q` (title search) |
| `GET /clusters/{id}` | one cluster with all member issues |
| `GET /issues` | all issues; filters: `beginner`, `q` |

## Roadmap (Phase 2)

- Learner skill profiles + matching score
- Auto-generated "first steps" plan per issue
- L2A project-intake integration
