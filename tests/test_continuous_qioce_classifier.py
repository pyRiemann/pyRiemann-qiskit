import numpy as np
import pytest

from pyriemann_qiskit.classification.continuous_qioce_classifier import ContinuousQIOCEClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    """Tiny separable binary dataset: class 0 = zeros, class 1 = ones."""
    rng = np.random.RandomState(42)
    X0 = rng.uniform(0.0, 0.1, (4, 2))   # cluster near 0
    X1 = rng.uniform(0.9, 1.0, (4, 2))   # cluster near 1
    X = np.vstack([X0, X1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(binary_data):
    """Fit a minimal classifier once; reused by all tests in the module."""
    X, y = binary_data
    clf = ContinuousQIOCEClassifier(n_reps=1, max_features=5)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Pure-numpy tests (no quantum, fast)
# ---------------------------------------------------------------------------

class TestNormalizeFeatures:
    def test_output_range(self):
        clf = ContinuousQIOCEClassifier()
        X = np.array([[0.0, 2.0], [1.0, 4.0], [0.5, 3.0]])
        X_norm = clf._normalize_features(X)
        assert X_norm.min() >= 0.0
        assert X_norm.max() <= np.pi + 1e-9

    def test_constant_column_no_division_by_zero(self):
        clf = ContinuousQIOCEClassifier()
        X = np.array([[1.0, 1.0], [1.0, 2.0]])  # first column constant
        X_norm = clf._normalize_features(X)
        assert np.all(np.isfinite(X_norm))

    def test_shape_preserved(self):
        clf = ContinuousQIOCEClassifier()
        X = np.random.rand(10, 3)
        assert clf._normalize_features(X).shape == X.shape


class TestErrorCases:
    def test_multiclass_raises(self):
        X = np.random.rand(6, 2)
        y = np.array([0, 1, 2, 0, 1, 2])
        with pytest.raises(ValueError, match="binary"):
            ContinuousQIOCEClassifier(n_reps=1).fit(X, y)

    def test_max_features_exceeded_raises(self):
        X = np.random.rand(4, 6)
        y = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="max_features"):
            ContinuousQIOCEClassifier(n_reps=1, max_features=4).fit(X, y)

    def test_predict_before_fit_raises(self):
        clf = ContinuousQIOCEClassifier()
        with pytest.raises(ValueError, match="fit"):
            clf.predict(np.random.rand(2, 2))

    def test_predict_proba_before_fit_raises(self):
        clf = ContinuousQIOCEClassifier()
        with pytest.raises(ValueError, match="fit"):
            clf.predict_proba(np.random.rand(2, 2))


# ---------------------------------------------------------------------------
# Post-fit attribute tests
# ---------------------------------------------------------------------------

class TestFitAttributes:
    def test_classes_set(self, fitted_clf):
        assert hasattr(fitted_clf, "classes_")
        np.testing.assert_array_equal(fitted_clf.classes_, [0, 1])

    def test_n_features_set(self, fitted_clf, binary_data):
        X, _ = binary_data
        assert fitted_clf.n_features_ == X.shape[1]

    def test_optim_params_set(self, fitted_clf):
        assert hasattr(fitted_clf, "optim_params_")
        assert len(fitted_clf.optim_params_) > 0

    def test_training_loss_history_non_empty(self, fitted_clf):
        assert hasattr(fitted_clf, "training_loss_history_")
        assert len(fitted_clf.training_loss_history_) > 0

    def test_training_loss_history_finite(self, fitted_clf):
        assert np.all(np.isfinite(fitted_clf.training_loss_history_))

    def test_x_min_x_max_set(self, fitted_clf, binary_data):
        X, _ = binary_data
        assert hasattr(fitted_clf, "X_min_")
        assert hasattr(fitted_clf, "X_max_")
        assert fitted_clf.X_min_.shape == (X.shape[1],)
        assert fitted_clf.X_max_.shape == (X.shape[1],)

    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = ContinuousQIOCEClassifier(n_reps=1, max_features=5)
        ret = clf.fit(X, y)
        assert ret is clf


# ---------------------------------------------------------------------------
# Predict / predict_proba / score tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_clf, binary_data):
        X, _ = binary_data
        y_pred = fitted_clf.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_predict_labels_from_classes(self, fitted_clf, binary_data):
        X, _ = binary_data
        y_pred = fitted_clf.predict(X)
        assert set(y_pred).issubset(set(fitted_clf.classes_))

    def test_predict_proba_shape(self, fitted_clf, binary_data):
        X, _ = binary_data
        proba = fitted_clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_sums_to_one(self, fitted_clf, binary_data):
        X, _ = binary_data
        proba = fitted_clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_proba_in_range(self, fitted_clf, binary_data):
        X, _ = binary_data
        proba = fitted_clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_score_in_range(self, fitted_clf, binary_data):
        X, y = binary_data
        s = fitted_clf.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_predict_consistent_with_proba(self, fitted_clf, binary_data):
        """predict labels match argmax of predict_proba."""
        X, _ = binary_data
        proba = fitted_clf.predict_proba(X)
        y_pred = fitted_clf.predict(X)
        expected = fitted_clf.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(y_pred, expected)
