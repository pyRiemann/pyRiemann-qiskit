from sklearn.base import TransformerMixin


class NoDimRed(TransformerMixin):
    """No dimensional reduction.

    A Ghost transformer that just returns the data without
    transformation.
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
