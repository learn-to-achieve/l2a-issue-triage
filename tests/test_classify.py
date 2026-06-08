"""Unit tests for src.classify parsing/clamping — does NOT call the API.
Run: python -m unittest tests.test_classify"""

import unittest

from src import classify as c


class TestParseResponse(unittest.TestCase):
    def test_plain_json(self):
        out = c.parse_response('{"type":"bug","difficulty":"beginner","confidence":0.9,"rationale":"x"}')
        self.assertEqual(out["type"], "bug")
        self.assertEqual(out["difficulty"], "beginner")
        self.assertFalse(out["needs_review"])

    def test_fenced_json(self):
        text = '```json\n{"type":"feature","difficulty":"advanced","confidence":0.8,"rationale":"y"}\n```'
        out = c.parse_response(text)
        self.assertEqual(out["type"], "feature")
        self.assertEqual(out["difficulty"], "advanced")

    def test_chatty_wrapper(self):
        text = 'Sure! Here is the classification:\n{"type":"docs","difficulty":"beginner","confidence":0.7}'
        self.assertEqual(c.parse_response(text)["type"], "docs")

    def test_garbage_falls_back(self):
        out = c.parse_response("I cannot help with that.")
        self.assertEqual(out, dict(c.FALLBACK))
        self.assertTrue(out["needs_review"])

    def test_empty_falls_back(self):
        self.assertEqual(c.parse_response(""), dict(c.FALLBACK))


class TestCoerce(unittest.TestCase):
    def test_hallucinated_labels_clamped(self):
        out = c.coerce({"type": "regression", "difficulty": "trivial", "confidence": 0.9})
        self.assertEqual(out["type"], "other")          # not in vocab -> other
        self.assertEqual(out["difficulty"], "intermediate")

    def test_confidence_out_of_range_and_nonnumeric(self):
        self.assertEqual(c.coerce({"type": "bug", "confidence": 5})["confidence"], 1.0)
        self.assertEqual(c.coerce({"type": "bug", "confidence": "high"})["confidence"], 0.0)

    def test_low_confidence_flags_review(self):
        self.assertTrue(c.coerce({"type": "bug", "confidence": 0.2})["needs_review"])
        self.assertFalse(c.coerce({"type": "bug", "confidence": 0.8})["needs_review"])


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeChat:
    """Returns a fixed reply for any prompt — lets us test roles offline."""
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return _Resp(self._content)


class TestVerifier(unittest.TestCase):
    rec = {"number": 1, "title": "t", "clean_text": "body"}
    label = {"type": "bug", "difficulty": "beginner", "confidence": 0.6,
             "rationale": "x", "needs_review": False}

    def test_verifier_agrees(self):
        chat = _FakeChat('{"agree": true, "type": "bug", "difficulty": "beginner", "reason": "ok"}')
        out = c._verify_one(chat, self.rec, self.label)
        self.assertTrue(out["agree"])

    def test_verifier_disagrees_on_changed_label(self):
        # says agree:true but changes the type -> treated as a disagreement
        chat = _FakeChat('{"agree": true, "type": "feature", "difficulty": "beginner", "reason": "req"}')
        out = c._verify_one(chat, self.rec, self.label)
        self.assertFalse(out["agree"])
        self.assertEqual(out["type"], "feature")

    def test_verifier_fails_open_when_unavailable(self):
        class Dead:
            def invoke(self, p): raise RuntimeError("boom")
        out = c._verify_one(Dead(), self.rec, self.label)
        self.assertTrue(out["agree"])  # never invents a disagreement


class TestRouter(unittest.TestCase):
    def _fixture(self):
        records = [{"number": i, "title": f"issue {i}"} for i in range(4)]
        clusters = [[0], [1, 2], [3]]                      # sizes 1, 2, 1
        labels = [
            {"type": "bug", "difficulty": "beginner", "needs_review": False},
            {"type": "docs", "difficulty": "beginner", "needs_review": False},
            {"type": "bug", "difficulty": "advanced", "needs_review": False},
        ]
        return records, clusters, labels

    def test_route_deterministic_picks_largest_matching(self):
        records, clusters, labels = self._fixture()
        out = c.route({"difficulty": "beginner"}, records, clusters, labels)  # no chat
        self.assertEqual(out["cluster_id"], 1)             # the size-2 beginner cluster

    def test_route_no_match(self):
        records, clusters, labels = self._fixture()
        out = c.route({"difficulty": "intermediate"}, records, clusters, labels)
        self.assertIsNone(out["cluster_id"])

    def test_route_llm_pick_clamped_to_candidate(self):
        records, clusters, labels = self._fixture()
        chat = _FakeChat('{"cluster_id": 999, "reason": "hallucinated"}')  # invalid id
        out = c.route({"skill": "css", "difficulty": "beginner"}, records, clusters, labels, chat=chat)
        self.assertIn(out["cluster_id"], {0, 1})           # clamped to a real candidate


if __name__ == "__main__":
    unittest.main()
