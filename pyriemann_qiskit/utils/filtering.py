from sklearn.base import TransformerMixin


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


class NaiveEvenDimRed(TransformerMixin):
    """Naive dimensional reduction (Even)

    Reduce the dimension of the feature by two,
    keeping only even indices of the feature vector.

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def fit(self, X, y=None):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples, n_features)
            Target vector relative to X.

        Returns
        -------
        self : NaiveEvenDimRed instance
            The NaiveEvenDimRed instance.
        """
        return self

    def transform(self, X):
        """Divide the feature dimension by two,
        keeping only even indices.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            ndarray of feature vectors.

        Returns
        -------
        X : ndarray, shape (n_samples, (n_features + 1) // 2)
            The Filtered feature vectors.
        """
        return X[:, ::2]


class NaiveOddDimRed(TransformerMixin):
    """Naive dimensional reduction (Odd)

    Reduce the dimension of the feature by two,
    keeping only odd indices of the feature vector.

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def fit(self, X, y=None):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : NaiveOddDimRed instance
            The NaiveOddDimRed instance.
        """
        return self

    def transform(self, X):
        """Divide the feature dimension by two,
        keeping only odd indices.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            ndarray of feature vectors.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features // 2)
            The Filtered feature vectors.
        """
        return X[:, 1::2]
