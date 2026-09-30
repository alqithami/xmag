#!/usr/bin/env python3
"""Regression checks for matched realized empirical false-positive rates."""
import math
import unittest
import numpy as np
from exact_fpr import empirical_point, tie_keys


class ActualFPRTests(unittest.TestCase):
    def test_distinct_tie_keys_and_reproducibility(self):
        a = tie_keys(10000)
        self.assertEqual(len(np.unique(a)), len(a))
        np.testing.assert_array_equal(a, tie_keys(10000))
        self.assertEqual(len(np.intersect1d(a, tie_keys(10000, 10000))), 0)

    def test_identical_actual_fpr_across_different_score_models(self):
        rng = np.random.default_rng(412)
        for n in [401, 1000, 143321]:
            for target in [.001, .01, .05]:
                expected_fp = math.floor(target * n)
                values = []
                for m in range(5):
                    negative = rng.integers(0, 3 + 4 * m, n).astype(float)
                    positive = rng.integers(0, 3 + 4 * m, 107).astype(float)
                    r = empirical_point(negative, positive, target)
                    self.assertEqual(r['empirical_false_positive_count'], expected_fp)
                    values.append(r['empirical_fpr'])
                self.assertEqual(len(set(values)), 1)
                self.assertEqual(values[0], expected_fp / n)

    def test_all_tied_scores_give_real_count_not_interpolated_count(self):
        r = empirical_point(np.ones(1000), np.ones(300), .01)
        self.assertEqual(r['empirical_false_positive_count'], 10)
        self.assertEqual(r['empirical_fpr'], .01)
        self.assertEqual(r['empirical_true_positive_count'], round(r['empirical_recall'] * 300))
        self.assertEqual(r, empirical_point(np.ones(1000), np.ones(300), .01))

    def test_vertical_roc_segment_without_score_ties(self):
        r = empirical_point(np.array([3., 2., 1., 0.]), np.array([2.5, 1.5]), .25)
        self.assertEqual(r['empirical_fpr'], .25)
        self.assertEqual(r['empirical_recall'], .5)

    def test_finite_sample_zero_budget(self):
        r = empirical_point(np.arange(100.), np.array([101., 102., 90.]), .001)
        self.assertEqual(r['empirical_false_positive_count'], 0)
        self.assertEqual(r['empirical_fpr'], 0)
        self.assertAlmostEqual(r['empirical_recall'], 2 / 3)

    def test_no_optimistic_score_selection(self):
        negative = np.arange(100.)
        a = empirical_point(negative, np.array([200., 201.]), .05)
        b = empirical_point(negative, np.array([-200., -201.]), .05)
        self.assertEqual(a['empirical_score_boundary'], b['empirical_score_boundary'])
        self.assertEqual(a['empirical_tie_key_hex'], b['empirical_tie_key_hex'])
        self.assertEqual(a['empirical_recall'], 1)
        self.assertEqual(b['empirical_recall'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
