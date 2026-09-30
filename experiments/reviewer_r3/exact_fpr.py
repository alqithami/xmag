#!/usr/bin/env python3
"""Identical *realized* empirical FPR by a fixed label-blind tie ordering.

Score is primary; a predeclared pseudorandom key is secondary. Each method
gets the same secondary key for the same ordered test observation. Only the
negative test scores select the retrospective operating boundary. The code
also retains ordinary strict/inclusive and randomized-expected ROC points.
"""
from __future__ import annotations
import math
import numpy as np

TIE_SEED = 20260930


def tie_keys(n, offset=0):
    # SplitMix64 is a bijection on uint64: distinct positions have distinct keys.
    z = np.arange(offset, offset + n, dtype=np.uint64) + np.uint64(TIE_SEED)
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def empirical_point(negative, positive, target):
    neg, pos = np.asarray(negative, dtype=np.float64), np.asarray(positive, dtype=np.float64)
    if neg.ndim != 1 or pos.ndim != 1 or not len(neg) or not len(pos):
        raise ValueError('Expected nonempty one-dimensional score populations.')
    if not np.isfinite(neg).all() or not np.isfinite(pos).all() or not 0 < target < 1:
        raise ValueError('Nonfinite scores or invalid FPR target.')
    budget = target * len(neg)
    if abs(budget - round(budget)) < 1e-10:
        budget = float(round(budget))
    m = int(math.floor(budget))
    # The next negative after m permitted false positives identifies the boundary.
    t = float(np.partition(neg, len(neg) - 1 - m)[len(neg) - 1 - m])
    above = int(np.count_nonzero(neg > t))
    left = m - above
    eq_indices = np.flatnonzero(neg == t)
    keys_neg = tie_keys(len(neg))
    keys_pos = tie_keys(len(pos), offset=len(neg))
    eq_keys = keys_neg[eq_indices]
    if not 0 <= left < len(eq_keys):
        raise AssertionError('Invalid lexicographic boundary.')
    key = np.partition(eq_keys, len(eq_keys) - 1 - left)[len(eq_keys) - 1 - left]
    reject_neg = (neg > t) | ((neg == t) & (keys_neg > key))
    reject_pos = (pos > t) | ((pos == t) & (keys_pos > key))
    if int(reject_neg.sum()) != m:
        raise AssertionError('Realized false-positive count differs from matched budget.')
    return {'empirical_fpr': m / len(neg), 'empirical_recall': float(reject_pos.mean()),
            'empirical_false_positive_count': m, 'empirical_true_positive_count': int(reject_pos.sum()),
            'empirical_negative_count': len(neg), 'empirical_unknown_count': len(pos),
            'empirical_score_boundary': t, 'empirical_tie_key_hex': hex(int(key)),
            'empirical_tie_seed': TIE_SEED,
            'empirical_tie_policy': 'score descending then fixed SplitMix64 row-position key descending; identical keys across methods within each task/denominator',
            'empirical_interpretation': 'retrospective test-ROC rule; realized FPR=floor(target*n_negative)/n_negative; not independent deployment calibration'}
