"""
Ensemble classifiers.
"""
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class JudgeClassifier(BaseEstimator, ClassifierMixin):

    """Judge classifier

    On this classifier implementation:

    "We trained both the quantum and classical algorithms
    on the balanced dataset[...].
    When the two classifiers disagreed on the label of a given transaction
    in the training set, the transaction was noted.
    These transactions, a subset of the training data of the balanced dataset,
    formed an additional dataset on which a metaclassifier was subsequently
    trained" [1]_.

    Parameters
    ----------
    c1 : ClassifierMixin
        An instance of ClassifierMixin.
    c2 : ClassifierMixin
        An instance of ClassifierMixin.
    judge : ClassifierMixin
        An instance of ClassifierMixin.
        This classifier is trained on the labels for which
        c1 and c2 obtain different predictions.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] M. Grossi et al.,
        ‘Mixed Quantum–Classical Method for Fraud Detection With Quantum
        Feature Selection’,
        IEEE Transactions on Quantum Engineering,
        doi: 10.1109/TQE.2022.3213474.

    """

    def __init__(self, c1, c2, judge):
        self.c1 = c1
        self.c2 = c2
        self.judge = judge

    def fit(self, X, y):
        """Train c1 and c2.
        Then Train the judge classifier on the the samples for which
        c1 and c2 have different predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) | (n_samples, n_features, n_times)
            The shape of X (vectors or matrices) should be the same for all classifiers
            (c1, c2 and judge).
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : JudgeClassifier instance
            The JudgeClassifier instance.
        """
        self.classes_ = np.unique(y)
        y1 = self.c1.fit(X, y).predict(X)
        y2 = self.c2.fit(X, y).predict(X)
        mask = np.not_equal(y1, y2)
        if not mask.all():
            self.judge.fit(X, y)
        else:
            y_diff = y[mask]
            X_diff = X[mask]
            self.judge.fit(X_diff, y_diff)

    def predict(self, X):
        """Calculates the predictions.
        When c1 and c2 don't have the same prediction,
        the judge classifier is used.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) | (n_samples, n_features, n_times)
            Input vectors or matrices.
            The shape of X should be the same for all classifiers.

        Returns
        -------
        pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        y1 = self.c1.predict(X)
        y2 = self.c2.predict(X)
        y_pred = y1
        mask = np.not_equal(y1, y2)
        if not mask.all():
            return y_pred
        X_diff = X[mask]
        y_pred[mask] = self.judge.predict(X_diff)
        return y_pred

    def predict_proba(self, X):
        """Return the probabilities associated with predictions.

        When c1 and c2 have the same prediction, the
        returned probability is the average of the probability of c1 and c2.

        When c1 and c2 don't have the same predictions,
        the returned probability is the the one of the judge classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) | (n_samples, n_features, n_times)
            Input vectors or matrices.
            The shape of X should be the same for all classifiers.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            The probability of the samples for each class in the model
        """
        y1_proba = self.c1.predict_proba(X)
        y2_proba = self.c2.predict_proba(X)
        y1 = self.c1.predict(X)
        y2 = self.c2.predict(X)
        predict_proba = (y1_proba + y2_proba) / 2
        mask = np.not_equal(y1, y2)
        if not mask.all():
            return predict_proba
        X_diff = X[mask]
        predict_proba[mask] = self.judge.predict_proba(X_diff)
        return predict_proba
