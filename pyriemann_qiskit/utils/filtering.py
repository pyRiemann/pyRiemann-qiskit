from sklearn.base import TransformerMixin


class NoDimRed(TransformerMixin):
    """No dimensional reduction.

    A Ghost transformer that just returns the data without
    transformation.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class NaiveEvenDimRed(TransformerMixin):
    """Naive dimensional reduction (Even)

    Reduce the dimension of the feature by two,
    keeping only even indices of the feature vector.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, ::2]


class NaiveOddDimRed(TransformerMixin):
    """Naive dimensional reduction (Odd)

    Reduce the dimension of the feature by two,
    keeping only odd indices of the feature vector.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, 1::2]
