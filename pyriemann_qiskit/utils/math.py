"""Module for mathematical helpers"""
from typing_extensions import deprecated
import numpy as np
from pyriemann.utils.covariance import normalize


@deprecated(
    "cov_to_corr_matrix is deprecated and will be removed in 0.4.0; "
    "please use pyriemann.utils.covariance.normalize."
)
def cov_to_corr_matrix(covmat):
    """Convert covariance matrices to correlation matrices.

    Parameters
    ----------
    covmat: ndarray, shape (..., n_channels, n_channels)
        Covariance matrices.

    Returns
    -------
    corrmat : ndarray, shape (..., n_channels, n_channels)
        Correlation matrices.

    Notes
    -----
    .. versionadded:: 0.0.2
    """
    return normalize(covmat, "corr")


def union_of_diff(*arrays):
    """Return the positions for which at least one of the array
    as a different value than the others.

    e.g.:

    A = 0 1 0
    B = 0 1 1
    C = 1 1 0

    return
    A = True False True

    Parameters
    ----------
    arrays: ndarray[], shape (n_samples,)[]
        A list of numpy arrays.

    Returns
    -------
    diff : ndarray, shape (n_samples,)
        A list of boolean.
        True at position i indicates that one of the array
        as a value different from the other ones at this
        position.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    size = len(arrays[0])
    for array in arrays:
        assert len(array) == size

    diff = [False] * size
    for i in range(size):
        s = set({array[i] for array in arrays})
        if len(s) > 1:
            diff[i] = True

    return np.array(diff)


def to_xyz(X):
    """Plot histogram of bi-class predictions.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2, 2)
        Array of SPD matrices of size 2 x 2.

    Returns
    -------
    points : ndarray, shape (n_matrices, 3)
        Cartesian representation in 3d.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    if X.ndim != 3:
        raise ValueError("Input `X` has not 3 dimensions")
    if X.shape[1] != 2 and X.shape[2] != 2:
        raise ValueError("SPD matrices must have size 2 x 2")
    return np.array([[spd[0, 0], spd[0, 1], spd[1, 1]] for spd in X])
