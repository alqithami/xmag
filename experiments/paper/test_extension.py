#!/usr/bin/env python3
"""Software regression checks only. Fixtures are not scientific measurements."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import core as c


class MetricTests(unittest.TestCase):
    def test_all_tied_scores_report_randomization_not_false_exact_threshold(self):
        r = c.fixed_fpr(np.ones(1000), np.ones(200), .01)
        self.assertAlmostEqual(r['expected_fpr'], .01)
        self.assertAlmostEqual(r['expected_recall'], .01)
        self.assertEqual(r['strict_fpr'], 0)
        self.assertEqual(r['inclusive_fpr'], 1)
        self.assertAlmostEqual(r['boundary_probability'], .01)

    def test_positive_only_vertical_segment_at_exact_fpr(self):
        r = c.fixed_fpr(np.array([3., 2., 1., 0.]), np.array([2.5, 1.5]), .25)
        self.assertEqual(r['threshold'], 2.)
        self.assertEqual(r['strict_fpr'], .25)
        self.assertEqual(r['strict_recall'], .5)
        self.assertEqual(r['expected_recall'], .5)

    def test_tied_boundary_with_both_populations(self):
        r = c.fixed_fpr(np.array([5., 4., 4., 1.]), np.array([6., 4., 0.]), .5)
        self.assertEqual(r['boundary_probability'], .5)
        self.assertAlmostEqual(r['expected_recall'], .5)
        self.assertEqual(r['strict_fpr'], .25)
        self.assertEqual(r['inclusive_fpr'], .75)

    def test_perfect_and_reversed_ranking(self):
        self.assertEqual(c.fixed_fpr(np.arange(100.), np.arange(101., 111.), .01)['expected_recall'], 1)
        self.assertEqual(c.fixed_fpr(np.arange(100.), np.arange(-11., -1.), .01)['expected_recall'], 0)

    def test_fixed_fpr_checks_all_three_levels(self):
        rng = np.random.default_rng(41)
        for target in c.LEVELS:
            for _ in range(10):
                r = c.fixed_fpr(rng.integers(0, 12, 401), rng.integers(0, 12, 107), target)
                self.assertAlmostEqual(r['expected_fpr'], target, places=13)
                self.assertLessEqual(r['strict_fpr'], target + 1e-12)
                self.assertGreaterEqual(r['inclusive_fpr'], target - 1e-12)

    def test_conformal_cutoff_exact_tied_rule(self):
        rng = np.random.default_rng(6)
        cal = rng.integers(0, 40, 2200).astype(float)
        score = np.r_[np.arange(-1., 42.), np.full(20, 39.)]
        for alpha in c.LEVELS:
            np.testing.assert_array_equal(c.conformal_p(cal, score) <= alpha,
                                          score > c.conformal_cutoff(cal, alpha))

    def test_packet_layouts_and_decoded_fields(self):
        f = {'class_id': np.array([1, 0]), 'feature_id': np.array([2, 1]),
             'probability': np.array([.812345, .33333]), 'contribution': np.array([-1.234567, .0023])}
        anomaly = np.array([.12531, .88342])
        for layout in ['12b', '16q', '24b']:
            M = c.packet_roundtrip(f, anomaly, layout, 3, 5)
            self.assertEqual(M.shape, (2, 4 if layout == '12b' else 9))
            expected = anomaly.astype(np.float16).astype(np.float32) if layout == '16q' else anomaly.astype(np.float32)
            np.testing.assert_array_equal(M[:, -1], expected)
        M = c.packet_roundtrip(f, anomaly, '16q', 3, 5)
        self.assertEqual(M[0, 3 + 2], np.float32(np.float16(f['contribution'][0])))
        self.assertEqual(np.count_nonzero(M[:, :3]), 2)

    def test_no_silent_packet_overflow(self):
        f = {'class_id': np.array([0]), 'feature_id': np.array([0]),
             'probability': np.array([1.0]), 'contribution': np.array([1e10])}
        with self.assertRaises(ValueError):
            c.packet_roundtrip(f, np.array([.1]), '16q', 3, 2)

    def test_score_monotonicity_and_range(self):
        rng = np.random.default_rng(11)
        a = rng.uniform(0, .5, (500, 3))
        b = a + rng.uniform(0, .5, (500, 3))
        for rule in ['composite', 'mean', 'max']:
            self.assertTrue(np.all(c.combine(a, rule) <= c.combine(b, rule)))
            self.assertTrue(np.all((c.combine(b, rule) >= 0) & (c.combine(b, rule) <= 1)))

    def test_full_small_fixture_and_resuming(self):
        """End-to-end fit, evaluation, quantization, packaging, and resume."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(201)
            n_each = 6000
            labels = np.repeat(['Benign', 'HTTPFlood', 'SlowrateDoS', 'UDPFlood'], n_each)
            n = len(labels)
            frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f'f{i}' for i in range(6)])
            frame['f0'] += np.repeat([0., 2., 3., 4.], n_each)
            frame['Attack Type'] = labels
            frame['Label'] = (labels != 'Benign').astype(int)
            frame['sVid'] = 'metadata-b'
            frame.loc[:1199, 'sVid'] = 'metadata-a'
            frame['dVid'] = 0
            frame['Attack Tool'] = labels
            frame['heldout_only_feature'] = (labels == 'UDPFlood').astype(int)
            csv = root / 'fixture.csv'
            frame.to_csv(csv, index=False)
            data = c.prepare(csv, 'UDPFlood', 7, fixture=True)
            self.assertNotIn('heldout_only_feature', data['raw_columns'])
            self.assertGreater(len(data['ids']['cal']), 1000)
            self.assertEqual(set(data['classes']), {'Benign', 'HTTPFlood', 'SlowrateDoS'})
            cmd = [sys.executable, str(Path(__file__).with_name('run.py')), '--root', str(root),
                   '--csv', str(csv), '--fixture', '--output-name', 'fixture_r3', '--jobs', '1']
            subprocess.run(cmd, check=True, timeout=300)
            summary = json.loads((root / 'results/fixture_r3/RUN_COMPLETENESS.json').read_text())
            self.assertEqual(summary['completed_tasks'], 1)
            self.assertEqual(summary['nominal_rows'], len(c.METHODS) * 3)
            self.assertEqual(summary['fixed_fpr_rows'], len(c.METHODS) * 6)
            self.assertEqual(summary['quantization_numerical_violations'], 0)
            marker = root / 'runs/fixture_r3/seed7/UDPFlood/COMPLETE.json'
            before = marker.read_bytes()
            subprocess.run(cmd, check=True, timeout=120)
            self.assertEqual(marker.read_bytes(), before)
            self.assertTrue((root / 'results/fixture_r3_review_text.txt').exists())
            self.assertTrue((root / 'results/fixture_r3_reviewer_results.zip').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
