"""Tests for the transfer-learning helpers."""

import numpy as np
import pytest
from pyriemann.classification import MDM
from pyriemann.datasets import make_matrices
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.utils.transfer import SelfAlignAdapter


@pytest.fixture
def subjects():
    """Three subjects with different channel mixings, two balanced classes."""
    rs = np.random.RandomState(42)
    n_trials, n_channels, n_times = 20, 4, 100
    X, y, groups = [], [], []
    for subject in range(3):
        mixing = make_matrices(1, n_channels, "spd", rs)[0]
        for label in (0, 1):
            source = rs.randn(n_trials, n_channels, n_times)
            source[:, label] *= 3.0
            X.append(np.einsum("ij,njt->nit", mixing, source))
            y += [label] * n_trials
            groups += [subject] * n_trials
    return np.concatenate(X), np.array(y), np.array(groups)


def make_adapter(scale=True):
    return SelfAlignAdapter(
        preprocessing=make_pipeline(Covariances(estimator="lwf")),
        estimator=MDM(),
        scale=scale,
    )


@pytest.mark.parametrize("scale", [True, False])
def test_align_recenters_to_identity(subjects, scale):
    """A subject aligned on its own trials has an identity Riemannian mean."""
    X, _, groups = subjects
    adapter = make_adapter(scale)
    covs = Covariances(estimator="lwf").fit_transform(X[groups == 0])
    aligned = adapter._align(covs)
    np.testing.assert_allclose(mean_riemann(aligned), np.eye(covs.shape[-1]), atol=1e-6)


def test_fit_predict_held_out_subject(subjects):
    """Train on two subjects, predict the third one as a single batch."""
    X, y, groups = subjects
    train, test = groups != 2, groups == 2
    adapter = make_adapter().fit(X[train], y[train], groups=groups[train])
    y_pred = adapter.predict(X[test])
    assert y_pred.shape == (test.sum(),)
    assert set(np.unique(y_pred)) <= set(adapter.classes_)
    proba = adapter.predict_proba(X[test])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_batch_order_does_not_matter(subjects):
    """The batch is aligned as a whole: reordering it reorders predictions."""
    X, y, groups = subjects
    train, test = groups != 2, groups == 2
    adapter = make_adapter().fit(X[train], y[train], groups=groups[train])
    perm = np.random.RandomState(0).permutation(test.sum())
    np.testing.assert_array_equal(
        adapter.predict(X[test][perm]), adapter.predict(X[test])[perm]
    )
