from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler


class NdRobustScaler(TransformerMixin):
    """Apply one robust scaler by feature.

    RobustScaler of scikit-learn [1]_ is adapted to 3d inputs [2]_.

    References
    ----------
    .. [1] \
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    .. [2] \
        https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix

    Notes
    -----
    .. versionadded:: 0.2.0
    """

    def __init__(self):
        self._scalers = []

    """Fits one robust scaler on each feature of the training data.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Training matrices.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NdRobustScaler instance
        The NdRobustScaler instance.
    """

    def fit(self, X, _y=None, **kwargs):
        _, n_features, _ = X.shape
        self._scalers = []
        for i in range(n_features):
            scaler = RobustScaler().fit(X[:, i, :])
            self._scalers.append(scaler)
        return self

    """Apply the previously trained robust scalers (on scaler by feature)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Matrices to scale.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NdRobustScaler instance
        The NdRobustScaler instance.
    """

    def transform(self, X, **kwargs):
        _, n_features, _ = X.shape
        if n_features != len(self._scalers):
            raise ValueError(
                "Input has not the same number of features as the fitted scaler"
            )
        for i in range(n_features):
            X[:, i, :] = self._scalers[i].transform(X[:, i, :])
        return X
