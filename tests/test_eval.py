"""Unit tests for eval.run_eval — run: python -m unittest tests.test_eval"""

import unittest

from eval import run_eval as e


class TestEval(unittest.TestCase):
    def test_load_golden_skips_comments_and_blanks(self):
        # The committed template has 5 data rows; # comments must be skipped.
        rows = e.load_golden()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all("cluster_id" in r for r in rows))

    def test_filled_excludes_todo_and_invalid(self):
        rows = [
            {"cluster_id": 0, "human_type": "TODO", "human_difficulty": "TODO"},
            {"cluster_id": 1, "human_type": "bug", "human_difficulty": "beginner"},
            {"cluster_id": 2, "human_type": "nonsense", "human_difficulty": "beginner"},
            {"cluster_id": 3, "human_type": "feature", "human_difficulty": "advanced"},
        ]
        kept = [r["cluster_id"] for r in e.filled(rows)]
        self.assertEqual(kept, [1, 3])

    def test_report_smoke(self):
        # sklearn metrics path runs without error on synthetic (NOT golden) labels.
        e._report("TYPE", ["bug", "feature", "bug"], ["bug", "bug", "bug"], e.TYPES)


if __name__ == "__main__":
    unittest.main()
