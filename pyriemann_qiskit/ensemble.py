"""Ensemble methods for quantum classifiers.

This module provides ensemble classification strategies that combine multiple
classifiers to improve prediction accuracy and robustness.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import union_of_diff


class JudgeClassifier(BaseEstimator, ClassifierMixin):

    """Judge classifier

    Several classifiers are trained on the balanced dataset.
    When at least one classifier disagrees on the label of a given input
    in the training set, the input was noted.
    These inputs, a subset of the training data of the balanced dataset,
    formed an additional dataset on which a new classifier was subsequently
    trained. Adapted from [1]_.

    Parameters
    ----------
    judge : ClassifierMixin
        Classifier trained on the inputs for which
        classifiers `clfs` obtain different predictions.

    clfs : list of ClassifierMixin
        Classifiers trained on the balanced dataset.

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

    def __init__(self, judge, clfs):
        self.clfs = clfs
        self.judge = judge

    def fit(self, X, y):
        """Fit all base classifiers and the judge classifier.

        Trains all classifiers in `clfs` on the full dataset, then identifies
        samples where classifiers disagree and trains the judge classifier on
        those samples only.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_features, n_times)
            Training data. Shape must be consistent across all classifiers.
        y : ndarray, shape (n_samples,)
            Target labels.

        Returns
        -------
        self : JudgeClassifier
            The fitted classifier instance.
        """
        self.classes_ = np.unique(y)
        ys = [clf.fit(X, y).predict(X) for clf in self.clfs]
        mask = union_of_diff(*ys)
        if not mask.any():
            self.judge.fit(X, y)
        else:
            self.judge.fit(X[mask], y[mask])

    def predict(self, X):
        """Predict class labels for samples in X.

        Uses base classifiers for prediction. When classifiers disagree on a
        sample, the judge classifier makes the final decision for that sample.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_features, n_times)
            Test samples. Shape must be consistent across all classifiers.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.

        See Also
        --------
        union_of_diff : Function to identify samples with disagreement
        """
        ys = [clf.predict(X) for clf in self.clfs]
        y_pred = ys[0]
        mask = union_of_diff(*ys)
        if not mask.any():
            return y_pred
        y_pred[mask] = self.judge.predict(X[mask])
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Returns averaged probabilities from base classifiers when they agree.
        When classifiers disagree, returns probabilities from the judge classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_features, n_times)
            Test samples. Shape must be consistent across all classifiers.

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        ys = [clf.predict_proba(X) for clf in self.clfs]
        predict_proba = sum(ys) / len(ys)
        mask = union_of_diff(*ys)
        if not mask.all():
            return predict_proba
        predict_proba[mask] = self.judge.predict_proba(X[mask])
        return predict_proba
