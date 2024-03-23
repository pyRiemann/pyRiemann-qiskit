import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm
from pyriemann_qiskit.utils.math import to_xyz
from scipy.spatial import ConvexHull

def plot_cvx_hull(X, ax):
    hull = ConvexHull(X)
    for simplex in hull.simplices:
        ax.plot(X[simplex, 0], X[simplex, 1], X[simplex, 2], 'k--', alpha=0.2)

def plot_manifold(X, y, plot_hull=False):
    if X.ndim != 3:
        raise ValueError("Input `covs` has not 3 dimensions")
    if X.shape[1] != 2 and X.shape[2] != 2:
        raise ValueError("SPD matrices must have size 2 x 2")

    classes = np.unique(y)

    points = to_xyz(X)
    points0 = points[y == classes[0]]
    points1 = points[y == classes[1]]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], alpha=1, color='blue')
    ax.scatter(points0[:, 0], points0[:, 1], points0[:, 2], alpha=0.5)

    if plot_hull:
        plot_cvx_hull(points, ax)
    return ax
