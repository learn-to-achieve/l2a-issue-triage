"""Unit tests for src.normalize — run: python -m unittest tests.test_normalize"""

import unittest

from src import normalize as n


class TestCleaning(unittest.TestCase):
    def test_strips_html_comments(self):
        body = "before <!-- template hint\nmultiline --> after"
        self.assertNotIn("template hint", n.clean_body(body))
        self.assertIn("before", n.clean_body(body))
        self.assertIn("after", n.clean_body(body))

    def test_strips_code_fences(self):
        body = "real text\n```python\nprint('boilerplate')\n```\nmore text"
        cleaned = n.clean_body(body)
        self.assertNotIn("boilerplate", cleaned)
        self.assertIn("real text", cleaned)
        self.assertIn("more text", cleaned)

    def test_strips_html_tags_and_collapses_ws(self):
        self.assertEqual(n.clean_body("<b>hi</b>   there"), "hi there")


class TestSignals(unittest.TestCase):
    def test_label_detection_beginner(self):
        issue = {"labels": [{"name": "good first issue"}, {"name": "type:bug"}]}
        labels = n.label_names(issue)
        self.assertIn("good first issue", labels)
        self.assertTrue(n.looks_beginner(labels))

    def test_label_detection_non_beginner(self):
        labels = n.label_names({"labels": [{"name": "type:bug"}, {"name": "P1"}]})
        self.assertFalse(n.looks_beginner(labels))

    def test_error_trace_detection_positive(self):
        body = 'Traceback (most recent call last):\n  File "x.py", line 3\nValueError: nope'
        self.assertTrue(n.has_error_trace(body))

    def test_error_trace_detection_negative(self):
        self.assertFalse(n.has_error_trace("Just a feature request, no stack."))

    def test_trace_detected_even_though_fence_stripped(self):
        # Trace lives in a code fence; detection runs on RAW body, cleaning after.
        raw = "Here is the error:\n```\nTraceback (most recent call last):\nKeyError: 'a'\n```"
        rec = n.normalize_one({"title": "boom", "body": raw, "number": 1})
        self.assertTrue(rec["has_error_trace"])
        self.assertNotIn("Traceback", rec["clean_text"])  # fence removed from text


class TestStaleness(unittest.TestCase):
    def setUp(self):
        from datetime import datetime, timezone
        self.now = datetime(2026, 6, 7, tzinfo=timezone.utc)

    def test_fresh_aging_stale(self):
        self.assertEqual(n.staleness("2026-06-01T00:00:00Z", self.now), "fresh")
        self.assertEqual(n.staleness("2026-03-01T00:00:00Z", self.now), "aging")
        self.assertEqual(n.staleness("2025-01-01T00:00:00Z", self.now), "stale")

    def test_missing_or_bad_date(self):
        self.assertEqual(n.staleness(None, self.now), "unknown")
        self.assertEqual(n.staleness("not-a-date", self.now), "unknown")


if __name__ == "__main__":
    unittest.main()
