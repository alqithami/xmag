#!/usr/bin/env python3
"""Regression checks for the supplied executed-snapshot adapter (not traffic results)."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.model_selection import train_test_split

import core as c
import reference
import snapshot_protocol as s


class FixedModel:
    classes_ = np.array(['a', 'b', 'c'])
    feature_importances_ = np.array([.1, .2, .7])
    def predict_proba(self, X):
        return np.tile([.5, .5, 0.], (len(X), 1))


class SnapshotTests(unittest.TestCase):
    def test_actual_validation_rule(self):
        for n in [20, 21, 1001, 150511]:
            ni, ci = s.validation_positions(n, 7)
            expected = np.random.default_rng(9024).permutation(n)
            np.testing.assert_array_equal(ni, expected[:n // 2])
            np.testing.assert_array_equal(ci, expected[n // 2:])
            self.assertEqual(len(set(ni).intersection(ci)), 0)

    def test_field_extraction_matches_executed_dense_proxy(self):
        X = np.array([[0, 0, 0], [1.001, -.04, 3.1415], [1e-43, 0, 0]], np.float32)
        data = {'X': {k: X for k in c.SPLITS}, 'owners': {k: np.zeros(len(X), int) for k in c.SPLITS},
                'classes': ['a', 'b', 'c']}
        m = FixedModel()
        got = s.source_fields(data, {0: m})['known']
        p = reference.owner_proba([m], X, np.zeros(len(X), int), data['classes'])
        phi = reference.owner_proxy([m], X, np.zeros(len(X), int), k=1)
        ci = np.argmax(p, axis=1)
        fi = np.argmax(abs(phi), axis=1)
        np.testing.assert_array_equal(got['class_id'], ci)
        np.testing.assert_array_equal(got['feature_id'], fi)
        np.testing.assert_array_equal(got['probability'], p[np.arange(len(X)), ci])
        np.testing.assert_array_equal(got['contribution'], phi[np.arange(len(X)), fi])
        self.assertEqual(int(got['class_id'][0]), 0)
        self.assertEqual(int(got['feature_id'][0]), 0)

    def test_prototype_float64_accumulation(self):
        rng = np.random.default_rng(17)
        M = rng.normal(size=(10001, 7)).astype(np.float32)
        y = np.where(np.arange(len(M)) % 2, 'a', 'b')
        mean, scale = s.fit_prototypes(M, y, ['a', 'b'])
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(scale.dtype, np.float64)
        np.testing.assert_array_equal(mean[0], M[y == 'a'].mean(axis=0, dtype=np.float64))
        np.testing.assert_array_equal(scale, M.std(axis=0, dtype=np.float64))

    def test_exact_zero_entropy(self):
        p = np.array([[1., 0., 0.], [.5, .5, 0.]])
        np.testing.assert_array_equal(s.entropy(p), -xlogy(p, p).sum(axis=1) / np.log(3))
        self.assertEqual(float(s.entropy(p)[0]), 0.)

    def test_verified_preparer_conversion(self):
        n = 101
        va = np.arange(10, 10 + n)
        pre = SimpleNamespace(feature_names_in_=np.array(['f0', 'f1']))
        d = SimpleNamespace(X_train=np.ones((10, 2), np.float32), y_train=np.array(['a'] * 10),
            a_train=np.zeros(10, int), idx_train=np.arange(10),
            X_val=np.arange(n * 2, dtype=np.float32).reshape(n, 2), y_val=np.array(['a'] * n),
            a_val=np.zeros(n, int), idx_val=va,
            X_known=np.ones((3, 2), np.float32), y_known=np.array(['a'] * 3), a_known=np.zeros(3, int), idx_known=np.arange(111, 114),
            X_unknown=np.ones((4, 2), np.float32), y_unknown=np.array(['z'] * 4), a_unknown=np.zeros(4, int), idx_unknown=np.arange(114, 118),
            classes=['a'], preprocessor=pre, feature_names=['f0', 'f1'], unknown_attack='z', agent_source='sVid')
        out = s.convert_prepared(d, 7)
        ni, ci = s.validation_positions(n, 7)
        np.testing.assert_array_equal(out['ids']['norm'], va[ni])
        np.testing.assert_array_equal(out['X']['cal'], d.X_val[ci])
        self.assertEqual(sum(out['class_counts'].values()), 118)

    def test_missing_actual_source_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                s.verify_and_load(Path(td))

    def test_supported_launcher_full_fixture_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(201)
            n_each = 6000
            labels = np.repeat(['Benign', 'HTTPFlood', 'SlowrateDoS', 'UDPFlood'], n_each)
            frame = pd.DataFrame(rng.normal(size=(len(labels), 6)), columns=[f'f{i}' for i in range(6)])
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
            check = s.prepare_fixture(csv, 'UDPFlood', 7, fixture=True)
            self.assertNotIn('heldout_only_feature', check['raw_columns'])
            known = np.flatnonzero(labels != 'UDPFlood')
            tv, _ = train_test_split(known, test_size=.3, random_state=7, stratify=labels[known])
            _, va = train_test_split(tv, test_size=.2, random_state=7, stratify=labels[tv])
            ni, ci = s.validation_positions(len(va), 7)
            np.testing.assert_array_equal(check['ids']['norm'], va[ni])
            np.testing.assert_array_equal(check['ids']['cal'], va[ci])
            cmd = [sys.executable, str(Path(__file__).with_name('start.py')), '--root', str(root),
                   '--csv', str(csv), '--fixture', '--output-name', 'fixture_snapshot', '--jobs', '1']
            subprocess.run(cmd, check=True, timeout=300)
            out = root / 'results' / 'fixture_snapshot'
            manifest = json.loads((out / 'RUN_COMPLETENESS.json').read_text())
            self.assertEqual(manifest['completed_tasks'], 1)
            self.assertEqual(manifest['quantization_numerical_violations'], 0)
            self.assertEqual(manifest['max_between_method_empirical_fpr_spread'], 0)
            plan = json.loads((out / 'PLAN.json').read_text())
            self.assertEqual(plan['version'], s.SNAPSHOT_VERSION)
            quant = pd.read_csv(out / 'quantization_diagnostics.csv')
            self.assertTrue((quant.calibration_analytic_max_score_bound + 1e-8 >= quant.calibration_max_score_change).all())
            self.assertTrue((quant.decision_change_bound_rate >= quant.decision_change_rate).all())
            marker = root / 'runs' / 'fixture_snapshot' / 'seed7' / 'UDPFlood' / 'COMPLETE.json'
            previous = marker.read_bytes()
            subprocess.run(cmd, check=True, timeout=120)
            self.assertEqual(marker.read_bytes(), previous)
            self.assertTrue((root / 'results' / 'fixture_snapshot_review_text.txt').is_file())
            self.assertTrue((root / 'results' / 'fixture_snapshot_reviewer_results.zip').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
