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
    y = np.array([0] * 9 + [1] * 3)  # 3:1 imbalance
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
    assert params == {
        "n_jobs": 4,
        "covariance": "cov",
        "estimator": "scm",
        "nfilter": 8,
        "delays": 4,
        "xdawn_estimator": "oas",
        "prototype_weight": None,
    }


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
# covariance definitions
# ---------------------------------------------------------------------------


def test_covariance_invalid(binary_data):
    X, y = binary_data
    with pytest.raises(ValueError, match="covariance must be"):
        QuantumStateDiscriminator(covariance="not_a_covariance").fit(X, y)


@pytest.mark.parametrize("covariance", ["cov", "erp", "hankel"])
def test_covariance_povm_stays_valid(binary_data, covariance):
    """Augmenting the operator must preserve the POVM guarantees."""
    X, y = binary_data
    clf = QuantumStateDiscriminator(covariance=covariance, nfilter=2).fit(X, y)

    # Looser than the full-rank case: an augmented rho_total is close to
    # singular, so inverting it amplifies floating-point error.
    n = clf.n_channels_
    np.testing.assert_allclose(sum(clf.povm_.values()), np.eye(n), atol=1e-6)
    for rho in clf.density_matrices_.values():
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(clf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_covariance_cov_is_default(binary_data):
    """Default must stay the plain covariance, unchanged by the new modes."""
    X, y = binary_data
    default = QuantumStateDiscriminator().fit(X, y)
    explicit = QuantumStateDiscriminator(covariance="cov").fit(X, y)
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))
    assert default.n_channels_ == X.shape[1]


def test_covariance_erp_augments_dimension(binary_data):
    """ERP operator is the prototypes stacked on the filtered trial."""
    X, y = binary_data
    clf = QuantumStateDiscriminator(covariance="erp", nfilter=2).fit(X, y)
    n_proto = clf.cov_estimator_.P_.shape[0]
    filtered = clf.cov_estimator_.Xd_.transform(X).shape[1]
    assert clf.n_channels_ == n_proto + filtered


def test_covariance_hankel_augments_dimension(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator(covariance="hankel", delays=3).fit(X, y)
    assert clf.n_channels_ == 3 * X.shape[1]


def test_prototype_weight_sets_energy_ratio(binary_data):
    """The weight is the prototype/trial energy ratio, not a raw scale."""
    X, y = binary_data
    weight = 5.0
    clf = QuantumStateDiscriminator(
        covariance="erp", nfilter=2, prototype_weight=weight
    ).fit(X, y)

    # After weighting, the prototype block should hold `weight` times the
    # energy of the average trial block.
    covmats = clf._operators(X)
    n_proto = clf.cov_estimator_.P_.shape[0]
    proto_energy = np.trace(covmats[0, :n_proto, :n_proto])
    trial_energy = np.mean(np.trace(covmats[:, n_proto:, n_proto:], axis1=-2, axis2=-1))
    np.testing.assert_allclose(proto_energy / trial_energy, weight, rtol=1e-8)


def test_prototype_weight_matches_scaling_the_prototype(binary_data):
    """The congruence equals scaling the prototype before estimating.

    Scaling the prototype rows by s multiplies the prototype block by s^2
    and the cross-blocks by s, which is what D.C.D does.
    """
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.utils.covariance import covariances_EP

    X, y = binary_data
    clf = QuantumStateDiscriminator(
        covariance="erp", nfilter=2, prototype_weight=5.0
    ).fit(X, y)

    xc = XdawnCovariances(nfilter=2, estimator="scm", xdawn_estimator="oas")
    xc.fit(X, y)
    scaled = covariances_EP(
        xc.Xd_.transform(X), xc.P_ * clf.prototype_scale_, estimator="scm"
    )
    np.testing.assert_allclose(clf._operators(X), scaled, rtol=1e-9)


def test_prototype_weight_none_leaves_prototype_unscaled(binary_data):
    X, y = binary_data
    clf = QuantumStateDiscriminator(covariance="erp", nfilter=2).fit(X, y)
    assert clf.prototype_scale_ == 1.0


def test_prototype_weight_ignored_without_erp(binary_data):
    """prototype_weight only applies to the erp operator."""
    X, y = binary_data
    plain = QuantumStateDiscriminator().fit(X, y)
    weighted = QuantumStateDiscriminator(prototype_weight=10).fit(X, y)
    np.testing.assert_array_equal(plain.predict(X), weighted.predict(X))


def test_covariance_erp_recovers_time_locked_signal():
    """A purely time-locked difference is invisible to a plain covariance.

    Both classes carry the same channel variance, so the classes differ
    only in *when* the deflection occurs. "cov" integrates over time and
    cannot separate them; "erp" keeps the prototype correlation and can.
    """
    rng = np.random.RandomState(0)
    n_trials, n_ch, n_times = 40, 4, 64

    X = rng.randn(n_trials * 2, n_ch, n_times) * 0.1
    deflection = np.hanning(16) * 3.0
    # Class 0: deflection early; class 1: same deflection, later.
    X[:n_trials, 0, 8:24] += deflection
    X[n_trials:, 0, 32:48] += deflection
    y = np.array([0] * n_trials + [1] * n_trials)

    cov = QuantumStateDiscriminator(covariance="cov").fit(X, y)
    erp = QuantumStateDiscriminator(covariance="erp", nfilter=2).fit(X, y)

    assert cov.score(X, y) < 0.7
    assert erp.score(X, y) > 0.95


# ---------------------------------------------------------------------------
# vectorization equivalence
# ---------------------------------------------------------------------------


def _naive_scores(clf, X):
    """Scores from the plain per-trial definition, as a reference.

    Normalizes each operator to unit trace and takes the Frobenius inner
    product with each POVM element, one trial at a time. The shipped code
    contracts the whole batch with einsum instead; this pins the two
    together so an optimisation cannot silently change the maths.
    """
    covmats = clf._operators(X)
    out = np.zeros((len(covmats), len(clf.classes_)))
    for i, C_i in enumerate(covmats):
        trace = np.trace(C_i)
        if trace < 1e-12:
            M = np.eye(len(C_i)) / len(C_i)
        else:
            M = C_i / trace
        for k, c in enumerate(clf.classes_):
            out[i, k] = np.sum(clf.povm_[c] * M)
    return out


@pytest.mark.parametrize("covariance", ["cov", "erp", "hankel"])
def test_scores_match_naive_definition(binary_data, covariance):
    X, y = binary_data
    clf = QuantumStateDiscriminator(covariance=covariance, nfilter=2).fit(X, y)
    np.testing.assert_allclose(
        clf.predict_proba(X), _naive_scores(clf, X), rtol=1e-9, atol=1e-12
    )


def test_scores_match_naive_definition_many_trials(rndstate):
    X = rndstate.randn(300, 4, 30)
    y = np.array([0, 1] * 150)
    clf = QuantumStateDiscriminator().fit(X, y)
    np.testing.assert_allclose(
        clf.predict_proba(X), _naive_scores(clf, X), rtol=1e-9, atol=1e-12
    )


def test_silent_trial_is_uninformative(binary_data):
    """An all-zero trial has no state: every class gets trace(Pi_c)/n."""
    X, y = binary_data
    clf = QuantumStateDiscriminator().fit(X, y)

    X_test = np.zeros((1, X.shape[1], X.shape[2]))
    proba = clf.predict_proba(X_test)
    expected = [np.trace(clf.povm_[c]) / clf.n_channels_ for c in clf.classes_]
    np.testing.assert_allclose(proba[0], expected, atol=1e-12)
    np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# n_jobs > 1
# ---------------------------------------------------------------------------


def test_parallel_predict_matches_serial(binary_data):
    X, y = binary_data
    clf_serial = QuantumStateDiscriminator(n_jobs=1).fit(X, y)
    clf_parallel = QuantumStateDiscriminator(n_jobs=2).fit(X, y)
    np.testing.assert_array_equal(clf_serial.predict(X), clf_parallel.predict(X))
