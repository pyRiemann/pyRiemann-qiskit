import numpy as np
from pyriemann.estimation import Covariances
from sklearn.base import TransformerMixin


class NoDimRed(TransformerMixin):
    """No dimensional reduction.

    A Ghost transformer that just returns the data without transformation.

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
    """Naive dimensional reduction.

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


class ChannelSelection(TransformerMixin):
    """Select channel in epochs.

    Select channels based on covariance information,
    keeping only channels with maximum covariances.

    Work on signal epochs.

    Parameters
    ----------
    n_channels : int
        The number of channels to select.
    cov_est : string, default="lwf"
        The covariance estimator.

    Notes
    -----
    .. versionadded:: 0.3.0
    """

    def __init__(self, n_channels, cov_est="lwf"):
        self.n_channels = n_channels
        self.cov_est = cov_est

    @staticmethod
    def _get_indices(maxes, mean_cov):
        indices = []
        for v in maxes:
            indices.extend(np.argwhere(mean_cov == v).flatten())
        return np.unique(indices)

    def fit(self, X, y=None, **kwargs):
        """Select channel based on covariances

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_features, n_samples)
            Training matrices.
        y : None | ndarray, shape (n_samples,), default=None
            Unused. Kept for scikit-learn compatibility.

        Returns
        -------
        self : ChannelSelection instance
            The ChannelSelection instance.
        """

        # Get the covariances of the channels for each epoch.
        covs = Covariances(estimator=self.cov_est).fit_transform(X)
        # Get the average covariance between the channels.
        mean_cov = np.mean(covs, axis=0)
        n_feats, _ = mean_cov.shape
        # Select the `n_channels` channels having the maximum covariances.
        all_max = []
        for i in range(n_feats):
            for j in range(i, n_feats):
                self._chs_idx = ChannelSelection._get_indices(all_max, mean_cov)

                if len(self._chs_idx) < self.n_channels:
                    all_max.append(mean_cov[i, j])
                else:
                    if mean_cov[i, j] > max(all_max):
                        all_max[np.argmin(all_max)] = mean_cov[i, j]

        return self

    def transform(self, X, **kwargs):
        """Select channels based on the computed covariance.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_features, n_samples)
            Input matrices.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_samples)
            Matrices with only the selected channel.
        """
        return X[:, self._chs_idx, :]
