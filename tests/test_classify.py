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


if __name__ == "__main__":
    unittest.main()
