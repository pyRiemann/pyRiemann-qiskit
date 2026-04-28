import numpy as np
from unittest.mock import patch
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from conftest import FixedPredClassifier

from pyriemann_qiskit.ensemble import JudgeClassifier


def test_canary():
    try:
        JudgeClassifier(SVC(), [SVC(), SVC()])
    except Exception:
        assert False


def test_get_set_params():
    X = np.array([[0] * 4, [0] * 4, [1] * 4, [1] * 4])
    y = np.array([0, 0, 1, 1])
    estimator = JudgeClassifier(SVC(), [SVC(), SVC()])
    skf = StratifiedKFold(n_splits=2)
    scr = cross_val_score(
        estimator, X, y, cv=skf, scoring="accuracy", error_score="raise"
    )
    assert scr.mean() > 0


def test_judge_required():
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])

    c1 = FixedPredClassifier([0, 1, 1])
    c2 = FixedPredClassifier([0, 0, 0])

    class Judge(ClassifierMixin):
        def fit(self, X, y):
            # Check that the two last samples were identified as different
            assert np.array_equal(X, [[0], [1]])
            assert np.array_equal(y, [0, 1])
            return self

        def predict(self, _X):
            # c1 and c2 will disagree on second and third predictions
            # return the true label of the second and third predictions
            return np.array([0, 1])

    estimator = JudgeClassifier(Judge(), clfs=[c1, c2])
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert np.array_equal(y, y_pred)


def test_judge_required_3_classifiers():
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])

    c1 = FixedPredClassifier([0, 1, 1, 1])
    c2 = FixedPredClassifier([0, 0, 0, 1])
    c3 = FixedPredClassifier([0, 1, 1, 0])

    class Judge(ClassifierMixin):
        def fit(self, X, y):
            # Check that the tree last samples were identified as different
            assert np.array_equal(X, [[0], [1], [1]])
            assert np.array_equal(y, [0, 1, 1])
            return self

        def predict(self, _X):
            # c1, c2 and c3 will disagree on 2nd, 3rd and 4th predictions
            # return the true label of these predictions.
            return np.array([0, 1, 1])

    estimator = JudgeClassifier(Judge(), clfs=[c1, c2, c3])
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert np.array_equal(y, y_pred)


def test_judge_not_required():
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])

    clf = FixedPredClassifier([0, 0, 1])

    class Judge(ClassifierMixin):
        def fit(self, X_judge, y_judge):
            # Check that the judge is fit on all the dataset
            # This is default behavior when the two classifiers agree
            assert np.array_equal(X, X_judge)
            assert np.array_equal(y, y_judge)

        def predict(self, _X):
            # Check the judge is never called on predict as the two classifiers agree
            assert False

    # Test with two same classifiers
    estimator = JudgeClassifier(Judge(), clfs=[clf, clf])
    estimator.fit(X, y)
    estimator.predict(X)

    # ... and three
    estimator = JudgeClassifier(Judge(), clfs=[clf, clf, clf])
    estimator.fit(X, y)
    estimator.predict(X)


# ---------------------------------------------------------------------------
# Regression: fit() must return self
# ---------------------------------------------------------------------------


def test_fit_returns_self_no_disagreement():
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([0, 0, 1])
    clf = FixedPredClassifier([0, 0, 1])
    estimator = JudgeClassifier(clf, [clf, clf])
    assert estimator.fit(X, y) is estimator


def test_fit_returns_self_with_disagreement():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    c1 = FixedPredClassifier([0, 1])
    c2 = FixedPredClassifier([1, 0])
    estimator = JudgeClassifier(c1, [c1, c2])
    assert estimator.fit(X, y) is estimator


# ---------------------------------------------------------------------------
# Regression: predict_proba early-return aligned with predict()
# ---------------------------------------------------------------------------


def _tracking_judge():
    """Returns (judge, call_log). call_log grows on each predict_proba call."""
    call_log = []

    class _TrackingJudge(ClassifierMixin):
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            call_log.append(X.copy())
            return np.full((len(X), 2), 0.5)

    return _TrackingJudge(), call_log


def test_predict_proba_judge_called_on_partial_disagreement():
    judge, call_log = _tracking_judge()
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    c1 = FixedPredClassifier([0, 0, 0])
    c2 = FixedPredClassifier([0, 1, 0])

    estimator = JudgeClassifier(judge, [c1, c2])
    estimator.fit(X, y)

    with patch(
        "pyriemann_qiskit.ensemble.union_of_diff",
        return_value=np.array([False, True, False]),
    ):
        estimator.predict_proba(X)

    assert len(call_log) == 1, "judge.predict_proba was not called"
    np.testing.assert_array_equal(call_log[0], [[1.0]])


def test_predict_proba_judge_not_called_when_all_agree():
    judge, call_log = _tracking_judge()
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    clf = FixedPredClassifier([0, 1])

    estimator = JudgeClassifier(judge, [clf, clf])
    estimator.fit(X, y)

    with patch(
        "pyriemann_qiskit.ensemble.union_of_diff",
        return_value=np.array([False, False]),
    ):
        estimator.predict_proba(X)

    assert len(call_log) == 0, "judge.predict_proba was called unexpectedly"
