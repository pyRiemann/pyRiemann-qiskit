"""Tests for QuantumStateMDM classifier."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuantumStateMDM


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
    clf = QuantumStateMDM()
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Fit / attributes
# ---------------------------------------------------------------------------


def test_fit_returns_self(binary_data):
    X, y = binary_data
    clf = QuantumStateMDM()
    assert clf.fit(X, y) is clf


def test_fit_attributes(fitted_clf, binary_data):
    X, y = binary_data
    assert hasattr(fitted_clf, "classes_")
    assert hasattr(fitted_clf, "density_matrices_")
    assert hasattr(fitted_clf, "n_channels_")
    assert hasattr(fitted_clf, "n_qubits_")
    assert hasattr(fitted_clf, "n_channels_padded_")
    assert fitted_clf.n_channels_ == X.shape[1]
    assert fitted_clf.n_channels_padded_ == 2 ** fitted_clf.n_qubits_
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
    clf = QuantumStateMDM().fit(X, y)
    for rho in clf.density_matrices_.values():
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


def test_density_matrices_trace_leq_one(binary_data):
    """Trace of each density matrix should be <= 1 (it equals 1 when no
    truncation happens, slightly less after the channel truncation)."""
    X, y = binary_data
    clf = QuantumStateMDM().fit(X, y)
    for rho in clf.density_matrices_.values():
        assert np.trace(rho) <= 1.0 + 1e-10


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
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_proba_shape(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(fitted_clf.classes_))


def test_predict_proba_sums_to_one(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_predict_proba_non_negative(fitted_clf, binary_data):
    X, _ = binary_data
    proba = fitted_clf.predict_proba(X)
    assert np.all(proba >= 0.0)


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
    rng = np.random.RandomState(0)
    X = rng.randn(6, 5, 40)  # 5 channels — not a power of 2
    y = np.array([0] * 3 + [1] * 3)
    clf = QuantumStateMDM().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (6,)
    assert clf.n_qubits_ == 3          # ceil(log2(5)) == 3
    assert clf.n_channels_padded_ == 8


def test_single_channel():
    rng = np.random.RandomState(7)
    X = rng.randn(4, 1, 20)
    y = np.array([0, 0, 1, 1])
    pred = QuantumStateMDM().fit(X, y).predict(X)
    assert pred.shape == (4,)


def test_multiclass():
    rng = np.random.RandomState(3)
    X = rng.randn(9, 4, 30)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    clf = QuantumStateMDM().fit(X, y)
    assert len(clf.classes_) == 3
    pred = clf.predict(X)
    assert pred.shape == (9,)
    proba = clf.predict_proba(X)
    assert proba.shape == (9, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_sklearn_clone():
    clf = QuantumStateMDM(n_jobs=2)
    clf2 = clone(clf)
    assert clf2.get_params() == clf.get_params()


def test_get_params():
    clf = QuantumStateMDM(n_jobs=4)
    params = clf.get_params()
    assert params == {"n_jobs": 4}


def test_sklearn_pipeline(binary_data):
    X, y = binary_data
    pipe = make_pipeline(QuantumStateMDM())
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# Separability sanity check
# ---------------------------------------------------------------------------


def test_separable_data():
    """On directionally separable data the classifier should achieve perfect accuracy.

    Amplitude encoding normalizes each time sample to unit norm, so sign
    differences cancel out (x and -x yield the same quantum state |ψ⟩⟨ψ|).
    We instead separate classes by which *channel* carries the signal.
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
    clf = QuantumStateMDM().fit(X, y)
    assert clf.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# n_jobs > 1
# ---------------------------------------------------------------------------


def test_parallel_predict_matches_serial(binary_data):
    X, y = binary_data
    clf_serial = QuantumStateMDM(n_jobs=1).fit(X, y)
    clf_parallel = QuantumStateMDM(n_jobs=2).fit(X, y)
    np.testing.assert_array_equal(clf_serial.predict(X), clf_parallel.predict(X))
