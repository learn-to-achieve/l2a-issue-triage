# L2A Issue Triage

AI-assisted triage agent that routes new open-source contributors to the right GitHub issue.

Built at [Learn to Achieve](https://l2a.me), a 501(c)(3) helping learners make real contributions from day one.

## Why

A new contributor faces hundreds of open issues: stale ones, duplicates, ones far beyond their level. They bounce around and contribute nothing. This is a triage problem — the same one production engineering teams face with crash reports.

## Architecture

```
GitHub API (multi-repo) --> ingest.py     (raw collection, cached)
                        --> normalize.py  (clean, extract signals)
                        --> cluster.py    (embeddings + FAISS dedupe)
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

## Roadmap (Phase 2)

- Learner skill profiles + matching score
- Auto-generated "first steps" plan per issue
- L2A project-intake integration
