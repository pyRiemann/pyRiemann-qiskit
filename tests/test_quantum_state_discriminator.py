"""Tests for QuantumStateDiscriminator."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuantumStateDiscriminator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rndstate():
    return np.random.RandomState(42)


@pytest.fixture(scope="module")
def binary_data(rndstate):
    X = rndstate.randn(10, 4, 50)
    y = np.array([0] * 5 + [1] * 5)
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator()
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Fit / attributes
# ---------------------------------------------------------------------------


def test_fit_returns_self(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator()
    assert clf.fit(X, y) is clf


def test_fit_attributes(fitted_clf, binary_data):
    X, y = binary_data
    assert hasattr(fitted_clf, "classes_")
    assert hasattr(fitted_clf, "density_matrices_")
    assert hasattr(fitted_clf, "povm_")
    assert hasattr(fitted_clf, "priors_")
    assert hasattr(fitted_clf, "n_channels_")
    assert fitted_clf.n_channels_ == X.shape[1]
    np.testing.assert_array_equal(fitted_clf.classes_, [0, 1])


def test_density_matrices_keys(fitted_clf):
    assert set(fitted_clf.density_matrices_.keys()) == {0, 1}


def test_density_matrices_shape(fitted_clf, binary_data):
    X, _ = binary_data
    n_ch = X.shape[1]
    for rho in fitted_clf.density_matrices_.values():
        assert rho.shape == (n_ch, n_ch)


def test_density_matrices_are_psd(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator().fit(X, y)
    for rho in clf.density_matrices_.values():
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


def test_density_matrices_trace_one(binary_data):
    """Each density matrix must have trace exactly 1."""
    X, y = binary_data
    clf = QuantumStateDiscriminator().fit(X, y)
    for rho in clf.density_matrices_.values():
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def test_priors_keys(fitted_clf):
    assert set(fitted_clf.priors_.keys()) == {0, 1}


def test_priors_sum_to_one(fitted_clf):
    total = sum(fitted_clf.priors_.values())
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_priors_match_class_frequencies(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator().fit(X, y)
    for c in clf.classes_:
        expected = np.mean(y == c)
        np.testing.assert_allclose(clf.priors_[c], expected, atol=1e-12)


def test_unequal_priors():
    """Classifier should handle imbalanced class distributions."""
    rng = np.random.RandomState(1)
    X = rng.randn(12, 4, 30)
    y = np.array([0] * 9 + [1] * 3)   # 3:1 imbalance
    clf = QuantumStateDiscriminator().fit(X, y)
    np.testing.assert_allclose(clf.priors_[0], 0.75, atol=1e-12)
    np.testing.assert_allclose(clf.priors_[1], 0.25, atol=1e-12)
    pred = clf.predict(X)
    assert pred.shape == (12,)


# ---------------------------------------------------------------------------
# POVM
# ---------------------------------------------------------------------------


def test_povm_keys(fitted_clf):
    assert set(fitted_clf.povm_.keys()) == {0, 1}


def test_povm_shape(fitted_clf, binary_data):
    X, _ = binary_data
    n_ch = X.shape[1]
    for Pi in fitted_clf.povm_.values():
        assert Pi.shape == (n_ch, n_ch)


def test_povm_completeness(fitted_clf, binary_data):
    """POVM elements must sum to identity: sum_c Pi_c = I."""
    X, _ = binary_data
    n_ch = X.shape[1]
    Pi_sum = sum(fitted_clf.povm_.values())
    np.testing.assert_allclose(Pi_sum, np.eye(n_ch), atol=1e-10)


def test_povm_elements_are_psd(fitted_clf):
    """Each POVM element must be positive semidefinite."""
    for Pi in fitted_clf.povm_.values():
        eigvals = np.linalg.eigvalsh(Pi)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# Predict shape / labels
# ---------------------------------------------------------------------------


def test_predict_shape(fitted_clf, binary_data):
    X, _ = binary_data
    pred = fitted_clf.predict(X)
    assert pred.shape == (X.shape[0],)


def test_predict_labels_from_classes(fitted_clf, binary_data):
    X, _ = binary_data
    pred = fitted_clf.predict(X)
    assert set(pred).issubset(set(fitted_clf.classes_))


# ---------------------------------------------------------------------------
# predict_proba — proper probabilities by construction (no softmax)
# ---------------------------------------------------------------------------


def test_predict_proba_shape(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(fitted_clf.classes_))


def test_predict_proba_sums_to_one(fitted_clf, binary_data):
    """POVM constraint guarantees exact sum-to-one."""
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_predict_proba_non_negative(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    assert np.all(proba >= -1e-10)


def test_predict_consistent_with_proba(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    y_pred = fitted_clf.predict(X)
    expected = fitted_clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(y_pred, expected)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_non_power_of_2_channels():
    """Channel count need not be a power of 2."""
    rng = np.random.RandomState(0)
    X = rng.randn(6, 5, 40)
    y = np.array([0] * 3 + [1] * 3)
    clf = QuantumStateDiscriminator().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (6,)
    assert clf.n_channels_ == 5


def test_single_channel():
    rng = np.random.RandomState(7)
    X = rng.randn(4, 1, 20)
    y = np.array([0, 0, 1, 1])
    pred = QuantumStateDiscriminator().fit(X, y).predict(X)
    assert pred.shape == (4,)


def test_multiclass():
    rng = np.random.RandomState(3)
    X = rng.randn(9, 4, 30)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    clf = QuantumStateDiscriminator().fit(X, y)
    assert len(clf.classes_) == 3
    pred = clf.predict(X)
    assert pred.shape == (9,)
    proba = clf.predict_proba(X)
    assert proba.shape == (9, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    # POVM completeness for 3 classes
    Pi_sum = sum(clf.povm_.values())
    np.testing.assert_allclose(Pi_sum, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_sklearn_clone():
    clf = QuantumStateDiscriminator(n_jobs=2)
    clf2 = clone(clf)
    assert clf2.get_params() == clf.get_params()


def test_get_params():
    clf = QuantumStateDiscriminator(n_jobs=4)
    params = clf.get_params()
    assert params == {"n_jobs": 4}


def test_sklearn_pipeline(binary_data):
    X, y = binary_data
    pipe = make_pipeline(QuantumStateDiscriminator())
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# Separability sanity check
# ---------------------------------------------------------------------------


def test_separable_data():
    """On directionally separable data the classifier should achieve
    perfect accuracy.

    Classes differ by which channel carries the signal, giving
    clearly distinct EEG covariance operators and density matrices.
    """
    rng = np.random.RandomState(0)
    n_trials, n_ch, n_times = 8, 4, 20
    # Class 0: energy concentrated on channel 0
    X0 = np.zeros((n_trials, n_ch, n_times))
    X0[:, 0, :] = rng.uniform(0.5, 1.0, (n_trials, n_times))
    # Class 1: energy concentrated on channel 1
    X1 = np.zeros((n_trials, n_ch, n_times))
    X1[:, 1, :] = rng.uniform(0.5, 1.0, (n_trials, n_times))
    X = np.concatenate([X0, X1], axis=0)
    y = np.array([0] * n_trials + [1] * n_trials)
    clf = QuantumStateDiscriminator().fit(X, y)
    assert clf.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# n_jobs > 1
# ---------------------------------------------------------------------------


def test_parallel_predict_matches_serial(binary_data):
    X, y = binary_data
    clf_serial = QuantumStateDiscriminator(n_jobs=1).fit(X, y)
    clf_parallel = QuantumStateDiscriminator(n_jobs=2).fit(X, y)
    np.testing.assert_array_equal(clf_serial.predict(X), clf_parallel.predict(X))
