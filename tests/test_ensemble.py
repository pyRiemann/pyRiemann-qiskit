from pyriemann_qiskit.ensemble import (
    JudgeClassifier,
)
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np


def test_canary():
    assert JudgeClassifier(SVC(), SVC(), SVC()) is not None


def test_get_set_params():
    X = np.array([[0] * 4, [0] * 4, [1] * 4, [1] * 4])
    y = np.array([0, 0, 1, 1])
    estimator = JudgeClassifier(SVC(), SVC(), SVC())
    skf = StratifiedKFold(n_splits=2)
    scr = cross_val_score(
        estimator, X, y, cv=skf, scoring="accuracy", error_score="raise"
    )
    assert scr.mean() > 0


def test_judge_required():
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])

    class C1(ClassifierMixin):
        def fit(self, _X, _y):
            return self

        def predict(self, _X):
            return np.array([0, 1, 1])

    class C2(ClassifierMixin):
        def fit(self, _X, _y):
            return self

        def predict(self, _X):
            return np.array([0, 0, 0])

    class Judge(ClassifierMixin):
        def fit(self, X, y):
            # Check that the two last samples were identified as different
            assert np.array_equal(X, [[0], [1]])
            assert np.array_equal(y, [0, 1])
            return self

        def predict(self, _X):
            # C1 and C2 will disagree on second and third predictions
            # return the true label of the second and third predictions
            return np.array([0, 1])

    estimator = JudgeClassifier(Judge(), clfs=[C1(), C2()])
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert np.array_equal(y, y_pred)


def test_judge_not_required():
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])

    class C1(ClassifierMixin):
        def fit(self, _X, _y):
            return self

        def predict(self, _X):
            return np.array([0, 0, 1])

    class C2(ClassifierMixin):
        def fit(self, _X, _y):
            return self

        def predict(self, _X):
            return np.array([0, 0, 1])

    class Judge(ClassifierMixin):
        def fit(self, X_judge, y_judge):
            # Check that the judge is fit on all the dataset
            # This is default behavior when the two classifiers agree
            assert np.array_equal(X, X_judge)
            assert np.array_equal(y, y_judge)

        def predict(self, _X):
            # Check the judge is never called on predict as the two classifiers agree
            assert False

    estimator = JudgeClassifier(Judge(), clfs=[C1(), C2()])
    estimator.fit(X, y)
    estimator.predict(X)
