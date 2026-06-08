"""Unit tests for src.ingest dedupe — run: python -m unittest tests.test_ingest"""

import unittest

from src import ingest


class TestMergeByNumber(unittest.TestCase):
    def test_dedupes_across_passes_keeping_first(self):
        newest = [{"number": 1, "title": "a"}, {"number": 2, "title": "b"}]
        beginner = [{"number": 2, "title": "b-dup"}, {"number": 3, "title": "c"}]
        merged = ingest._merge_by_number(newest, beginner)
        nums = sorted(i["number"] for i in merged)
        self.assertEqual(nums, [1, 2, 3])
        # First occurrence wins for the overlapping number.
        self.assertEqual(next(i for i in merged if i["number"] == 2)["title"], "b")

    def test_ignores_items_without_number(self):
        merged = ingest._merge_by_number([{"title": "no-number"}, {"number": 5}])
        self.assertEqual([i["number"] for i in merged], [5])

    def test_label_cache_path_differs(self):
        plain = ingest._cache_path("a/b")
        labeled = ingest._cache_path("a/b", "good first issue")
        self.assertNotEqual(plain, labeled)
        self.assertIn("label_good-first-issue", labeled.name)


if __name__ == "__main__":
    unittest.main()
