"""Module for mathematical helpers"""

import numpy as np


def cov_to_corr_matrix(covmat):
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    covmat: ndarray, shape (n_channels, n_channels)
        A covariance matrix.

    Returns
    -------
    corrmat : ndarray, shape (n_channels, n_channels)
        The correlation matrix.

    Notes
    -----
    .. versionadded:: 0.0.2
    """
    v = np.sqrt(np.diag(covmat))
    outer_v = np.outer(v, v)
    correlation = covmat / outer_v
    correlation[covmat == 0] = 0
    return correlation
