from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler


class NDRobustScaler(TransformerMixin):
    """Apply one robust scaler by feature.

    See [1]_ for more details.

    References
    ----------
    .. [1] \
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
    X : ndarray, shape (n_matrices, n_samples, n_features)
        Training matrices.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NDRobustScaler instance
        The NDRobustScaler instance.
    """

    def fit(self, X, _y=None, **kwargs):
        _, n_features, _ = X.shape
        self._scalers = []
        for i in range(n_features):
            scaler = RobustScaler()
            scaler.fit(X[:, i, :])
            self._scalers.append(scaler)
        return self

    """Apply the previously trained robust scalers (on scaler by feature)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_samples, n_features)
        Matrices to scale.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NDRobustScaler instance
        The NDRobustScaler instance.
    """

    def transform(self, X, **kwargs):
        n_features = len(self._scalers)
        for i in range(n_features):
            X[:, i, :] = self._scalers[i].transform(X[:, i, :])
        return X
