"""
Ensemble classifiers.
"""
import numpy as np
from pyriemann_qiskit.utils import union_of_diff
from sklearn.base import ClassifierMixin, BaseEstimator


class JudgeClassifier(BaseEstimator, ClassifierMixin):

    """Judge classifier

    Two (or more) classifiers are trained on the balanced dataset.
    When the two classifiers disagreed on the label of a given input
    in the training set, the input was noted.
    These inputs, a subset of the training data of the balanced dataset,
    formed an additional dataset on which a metaclassifier was subsequently
    trained. Adapted from [1]_.

    Parameters
    ----------
    clfs : ClassifierMixin[]
        An liast of ClassifierMixin.
    judge : ClassifierMixin
        An instance of ClassifierMixin.
        This classifier is trained on the labels for which
        clf1 and clf2 obtain different predictions.

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

    def __init__(self, judge, *clfs):
        self.clfs = clfs
        self.judge = judge

    def fit(self, X, y):
        """Train all classifiers in clfs.

        Then Train the judge classifier on the the samples for which
        the classifiers have different predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) | (n_samples, n_features, n_times)
            The shape of X (vectors or matrices) should be the same for all classifiers
            (clf1, clf2 and judge).
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : JudgeClassifier instance
            The JudgeClassifier instance.
        """
        self.classes_ = np.unique(y)
        ys = [clf.fit(X).predict(X) for clf in self.clfs]
        mask = union_of_diff(ys)
        if not mask.any():
            self.judge.fit(X, y)
        else:
            self.judge.fit(X[mask], y[mask])

    def predict(self, X):
        """Calculates the predictions.

        When the classifiers don't agree on the prediction
        the judge classifier is used.

        The behavior is that if at least one of the classifier
        doesn't have the same prediction for a particular sample,
        then this sample is passed over to the `judge`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) | (n_samples, n_features, n_times)
            Input vectors or matrices.
            The shape of X should be the same for all classifiers.

        Returns
        -------
        pred : array, shape (n_samples,)
            Class labels for samples in X.

        See also
        --------
        union_of_diff
        """
        ys = [clf.predict(X) for clf in self.clfs]
        y_pred = ys[0]
        mask = union_of_diff(ys)
        if not mask.any():
            return y_pred
        X_diff = X[mask]
        y_pred[mask] = self.judge.predict(X_diff)
        return y_pred

    def predict_proba(self, X):
        """Return the probabilities associated with predictions.

        When clf1 and clf2 have the same prediction, the
        returned probability is the average of the probability of clf1 and clf2.

        When clf1 and clf2 don't have the same predictions,
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
        ys = [clf.predict_proba(X) for clf in self.clfs]
        predict_proba = sum(ys) / len(ys)
        mask = union_of_diff(ys)
        if not mask.all():
            return predict_proba
        X_diff = X[mask]
        predict_proba[mask] = self.judge.predict_proba(X_diff)
        return predict_proba
