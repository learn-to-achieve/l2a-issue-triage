# Kubernetes manifests (illustrative)

> **Do NOT apply these to any cluster.** They are reference manifests showing how
> this app *would* run on Kubernetes. The actual live deployment is **GCP Cloud
> Run** (serverless, scale-to-zero) — see the root README.

- `deployment.yaml` — runs the read-only Streamlit board container (built from
  the repo-root `Dockerfile`): 1 replica, `containerPort: 8080`, readiness +
  liveness probes on `/`, and CPU/memory requests/limits.
- `service.yaml` — a `ClusterIP` Service fronting the Deployment (swap to
  `LoadBalancer` or add an Ingress for external access).

## Scaling the pipeline (note)

The board is a stateless reader and scales trivially (more replicas). The
expensive, embarrassingly-parallel part is **classification**: each cluster is
independent, so `classify` could run as **parallel Kubernetes Jobs** (or an
indexed Job / Job queue) — shard the clusters across workers, each writing its
labels, then merge into `triage.json`. The per-cluster caching and checkpointing
already in `classify.py` make that fan-out safe to retry.
