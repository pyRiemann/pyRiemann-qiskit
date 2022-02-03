from sklearn.base import TransformerMixin
import itertools
import numpy as np
from sklearn.svm import SVC

class Closest(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        print(indices, self.ndim)
        for v in X:
            for subset in itertools.combinations(indices, self.ndim):
                not_indices = [i for i in indices if i not in subset]
                sub_v = v.copy()
                sub_v[not_indices] = 0
                dist = np.linalg.norm(v - sub_v)
                key = ''.join(str(val) for val in subset)
                if key in ret:
                    ret[key] = ret[key] + dist
                else:
                    ret[key] = dist
        self.closest_subset_to_vector = min(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret


class Preclassif(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        for subset in itertools.combinations(indices, self.ndim):
            sub_vectors = None
            for v in X:
                if sub_vectors is None:
                    sub_vectors = np.atleast_2d([v[list(subset)]])
                else:
                    sub_vectors = np.append(sub_vectors, [v[list(subset)]], axis=0)
            svc = SVC()

            svc.fit(sub_vectors, y)
            key = ''.join(str(val) for val in subset)
            ret[key] = svc.score(sub_vectors, y)
        self.closest_subset_to_vector = max(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret



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
