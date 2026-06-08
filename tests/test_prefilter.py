"""Unit tests for src.prefilter — synthetic model, no data files needed (CI-safe).
Run: python -m unittest tests.test_prefilter"""

import unittest

from src import prefilter


def _synthetic_model():
    X = [
        "fix typo in the docs", "update the readme documentation", "correct spelling in the guide",
        "segfault crash null pointer", "crashes with a stack trace error", "throws an exception on load",
        "add dark mode feature request", "please add an export button", "feature: new dashboard view",
    ]
    y_type = ["docs", "docs", "docs", "bug", "bug", "bug", "feature", "feature", "feature"]
    y_diff = ["beginner", "beginner", "beginner",
              "intermediate", "intermediate", "intermediate",
              "intermediate", "intermediate", "intermediate"]
    return prefilter.build(X, y_type, y_diff)


class TestPrefilter(unittest.TestCase):
    def test_build_and_predict_label_shape(self):
        label, _ = prefilter.predict("there is a typo in the documentation", model=_synthetic_model())
        self.assertEqual(label["type"], "docs")
        self.assertEqual(label["source"], "prefilter")
        for k in ("type", "difficulty", "confidence", "rationale", "needs_review"):
            self.assertIn(k, label)

    def test_no_model_returns_none(self):
        # An empty/falsy model means "untrained" -> never short-circuits.
        self.assertEqual(prefilter.predict("anything", model={}), (None, False))

    def test_needs_review_tracks_confidence(self):
        label, confident = prefilter.predict("crash null pointer segfault", model=_synthetic_model())
        self.assertEqual(label["needs_review"], not confident)


if __name__ == "__main__":
    unittest.main()
