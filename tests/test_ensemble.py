from pyriemann_qiskit.ensemble import (
    JudgeClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score
import numpy as np


def test_canary():
    assert JudgeClassifier(LDA(), LDA(), LDA()) is not None


def test_get_set_params():
    X = np.array([[0], [0], [1]])
    y = np.array([0, 0, 1])
    estimator = JudgeClassifier(LDA(), LDA(), LDA())
    skf = StratifiedKFold(n_splits=2)
    scr = cross_val_score(
        estimator, X, y, cv=skf, scoring="roc_auc", error_score="raise"
    )
    assert scr.mean() > 0


def test_predict():
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
        def fit(self, _X, _y):
            return self

        def predict(self, _X):
            # C1 and C2 will disagree on second and third predictions
            # return the true label of the second and third predictions
            return [0, 1]

    estimator = JudgeClassifier(C1(), C2(), Judge())
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert np.array_equal(y, y_pred)
