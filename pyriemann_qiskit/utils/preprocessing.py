# Apply one scaler by channel:
# See Stackoverflow link for more details [4]
class NDRobustScaler(TransformerMixin):
    def __init__(self):
        self._scalers = []

    def fit(self, X, y=None, **kwargs):
        _, n_channels, _ = X.shape
        self._scalers = []
        for i in range(n_channels):
            scaler = RobustScaler()
            scaler.fit(X[:, i, :])
            self._scalers.append(scaler)
        return self

    def transform(self, X, **kwargs):
        n_channels = len(self._scalers)
        for i in range(n_channels):
            X[:, i, :] = self._scalers[i].transform(X[:, i, :])
        return X
