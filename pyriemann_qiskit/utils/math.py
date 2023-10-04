"""Module for mathematical helpers"""

from pyriemann.utils.covariance import normalize


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
