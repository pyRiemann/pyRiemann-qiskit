from pyriemann_qiskit.ensemble import (
    JudgeClassifier,
)
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np


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


class FixedPredClassifier(ClassifierMixin):
    def __init__(self, y_pred) -> None:
        self.y_pred = y_pred

    def fit(self, _X, _y):
        return self

    def predict(self, _X):
        return np.array(self.y_pred)


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
