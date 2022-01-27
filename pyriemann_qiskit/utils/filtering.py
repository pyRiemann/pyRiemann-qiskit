from sklearn.base import TransformerMixin


class NoDimRed(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class NaivePairDimRed(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, ::2]


class NaiveImpairDimRed(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, 1::2]
