"""Unit tests for api.main — injects a fixture so no data/triage.json is needed.
Run: python -m unittest tests.test_api"""

import unittest

from fastapi.testclient import TestClient

from api import main as api_main

_ISSUE_A = {"number": 1, "title": "cache bug on reload", "html_url": "http://x/1",
            "source_repo": "a/b", "labels": ["bug"], "looks_beginner": True,
            "has_error_trace": False, "staleness": "fresh"}
_ISSUE_B = {"number": 2, "title": "add export button", "html_url": "http://x/2",
            "source_repo": "a/b", "labels": [], "looks_beginner": False,
            "has_error_trace": False, "staleness": "stale"}
_CLUSTER = {"cluster_id": 0, "size": 2,
            "classification": {"type": "bug", "difficulty": "beginner",
                               "confidence": 0.9, "rationale": "r", "needs_review": False},
            "representative": _ISSUE_A, "issues": [_ISSUE_A, _ISSUE_B]}
_FIXTURE = {"summary": {"issues": 2, "clusters": 1, "beginner_issues": 1, "needs_review": 0},
            "clusters": [_CLUSTER]}


class TestAPI(unittest.TestCase):
    def setUp(self):
        api_main._data = lambda: _FIXTURE   # inject fixture (bypasses the file)
        self.client = TestClient(api_main.app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_clusters_filter_by_type(self):
        self.assertEqual(len(self.client.get("/clusters?type=bug").json()), 1)
        self.assertEqual(len(self.client.get("/clusters?type=feature").json()), 0)

    def test_clusters_filter_beginner(self):
        self.assertEqual(len(self.client.get("/clusters?beginner=true").json()), 1)

    def test_cluster_by_id(self):
        self.assertEqual(self.client.get("/clusters/0").json()["cluster_id"], 0)
        self.assertEqual(self.client.get("/clusters/999").status_code, 404)

    def test_issues_and_filter(self):
        self.assertEqual(len(self.client.get("/issues").json()), 2)
        self.assertEqual(len(self.client.get("/issues?beginner=true").json()), 1)
        self.assertEqual(len(self.client.get("/issues?q=export").json()), 1)


if __name__ == "__main__":
    unittest.main()
