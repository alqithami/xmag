from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mdpi_revision_common import (  # noqa: E402
    composite_scores,
    fit_coordinator,
    fit_fedavg_linear,
    serialize_message,
)


def test_message_serializers_have_declared_lengths():
    args = dict(
        class_id=1,
        prob=0.75,
        feature_id=7,
        contribution=0.125,
        anomaly=0.25,
        K=8,
        d=88,
    )
    assert len(serialize_message("full_16q", **args)) == 16
    assert len(serialize_message("full_20", **args)) == 20
    assert len(serialize_message("full_24", **args)) == 24


def test_composite_score_is_bounded_and_coordinatewise_monotone():
    ref = np.linspace(0.0, 1.0, 101)
    low = composite_scores(ref, ref, ref, np.array([0.2]), np.array([0.2]), np.array([0.2]))
    high_u = composite_scores(ref, ref, ref, np.array([0.8]), np.array([0.2]), np.array([0.2]))
    high_a = composite_scores(ref, ref, ref, np.array([0.2]), np.array([0.8]), np.array([0.2]))
    high_r = composite_scores(ref, ref, ref, np.array([0.2]), np.array([0.2]), np.array([0.8]))

    assert 0.0 <= low.item() <= 1.0
    assert 0.0 <= high_u.item() <= 1.0
    assert 0.0 <= high_a.item() <= 1.0
    assert 0.0 <= high_r.item() <= 1.0
    assert high_u.item() >= low.item()
    assert high_a.item() >= low.item()
    assert high_r.item() >= low.item()


def test_explicit_ovr_coordinator_supports_multiclass_data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(90, 6))
    y = np.asarray(["a", "b", "c"] * 30)
    model = fit_coordinator(X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    assert np.all(np.isfinite(proba))


def test_fedavg_regression_avoids_incremental_dtype_failure():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    y = np.asarray(["a", "b", "c"] * 60)
    owners = np.repeat(np.arange(3), 60)

    # Make the local partitions non-IID so some agents lack one class.
    y[(owners == 0) & (y == "c")] = "b"
    y[(owners == 1) & (y == "a")] = "b"

    data = SimpleNamespace(
        X_train=X,
        y_train=y,
        a_train=owners,
        classes=sorted(np.unique(y).tolist()),
    )
    model = fit_fedavg_linear(data, rounds=2, seed=7)
    proba = model.predict_proba(X[:10].astype(np.float64))

    assert proba.shape == (10, len(data.classes))
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
