from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class NoDimRed(TransformerMixin):
    """No dimensional reduction.

    A Ghost transformer that just returns the data without
    transformation.

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def fit(self, X, y=None):
        """Fit the training data.

        Parameters
        ----------
        X : object
            Training data.
            The type of the data does not matter,
            as the parameter will not be used.
        y : object
            Target vector relative to X.
            The type of the vector does not matter,
            as the parameter will not be used.

        Returns
        -------
        self : NoDimRed instance
            The NoDimRed instance.
        """
        return self

    def transform(self, X, y=None):
        """Transform the training data.

        No transformation will be applied in practice.
        This class implements only the Ghost design pattern.

        Parameters
        ----------
        X : object
            Training data.
            The type of the data does not matter.
        y : object
            Target vector relative to X.
            The type of the vector does not matter.

        Returns
        -------
        X : object
            The same data passed through the parameter X.
        """
        return X


class NaiveDimRed(TransformerMixin):
    """Naive dimensional reduction

    Reduce the dimension of the feature by two,

    Parameters
    ----------
    is_even : bool (default: True)
        - If true keep only even indices of feature vectors.
        - If false, keep only odd indices of feature vectors.

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def __init__(self, is_even=True):
        self.is_even = is_even

    def fit(self, X, y=None):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples,) (default: None)
            Target vector relative to X.
            In practice, never used.

        Returns
        -------
        self : NaiveDimRed instance
            The NaiveDimRed instance.
        """
        return self

    def transform(self, X):
        """Divide the feature dimension by two,
        keeping even (odd) indices if `is_even` is True (False).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            ndarray of feature vectors.

        Returns
        -------
        X : ndarray, shape (n_samples, (n_features + 1) // 2)
            The filtered feature vectors.
        """
        offset = 0 if self.is_even else 1
        return X[:, offset::2]


class Vectorizer(BaseEstimator, TransformerMixin):
    """This is an auxiliary transformer that allows one to vectorize data
       structures in a pipeline For instance, in the case of an X with
       dimensions Nt x Nc x Ns, one might be interested in a new data
       structure with dimensions Nt x (Nc.Ns)
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))
