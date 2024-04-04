"""
Visualization of the covariance matrices on the SPD manifold.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyriemann_qiskit.utils.math import to_xyz
from scipy.spatial import ConvexHull


def plot_cvx_hull(X, ax):
    """Plot the convex hull of a set of points.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 3)
        A set of 3d points in Cartesian coordinates.
    ax : Matplotlib.Axes
        A figure where to plot the points of the convex hull.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    hull = ConvexHull(X)
    for simplex in hull.simplices:
        ax.plot(X[simplex, 0], X[simplex, 1], X[simplex, 2], "k--", alpha=0.2)


def plot_manifold(X, y, plot_hull=False):
    """Plot spd matrices in 3d (cartesian coordinate system).

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2, 2)
        A set of SPD matrices of size 2 x 2.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    plot_hull : boolean, default=False
        If True, plot the convex hull of X.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    if X.ndim != 3:
        raise ValueError("Input `covs` has not 3 dimensions")
    if X.shape[1] != 2 and X.shape[2] != 2:
        raise ValueError("SPD matrices must have size 2 x 2")

    classes = np.unique(y)

    points = to_xyz(X)
    points0 = points[y == classes[0]]
    points1 = points[y == classes[1]]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        points0[:, 0], points0[:, 1], points0[:, 2], color="red", label=classes[0]
    )
    ax.scatter(
        points1[:, 0],
        points1[:, 1],
        points1[:, 2],
        alpha=0.5,
        color="blue",
        label=classes[1],
    )

    ax.legend(title="Classes", loc="upper center")

    if plot_hull:
        plot_cvx_hull(points, ax)
    return fig
