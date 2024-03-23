"""
Visualize distances (i.e. how well the classes are separable?)
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm
from pyriemann_qiskit.utils.math import to_xyz


def plot_bihist(X, y):
    """Plot histogram of bi-class predictions.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2)
        Predictions, distances or probabilities.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.

    Returns
    -------
    fig : matplotlib figure
        Figure of histogram.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    if X.ndim != 2:
        raise ValueError("Input X has not 2 dimensions")
    if X.shape[1] != 2:
        raise ValueError("Input X has not 2 classes")

    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise ValueError("Input y has not 2 classes")

    X = X / np.sum(X, axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.hist(X[y == classes[0], 0], label=classes[0], color="red")
    ax.hist(1 - X[y == classes[1], 1], label=classes[1], alpha=0.5, color="blue")
    ax.set(xlabel="Distances", ylabel="Frequency")
    ax.legend(title="Classes", loc="upper center")

    return fig


def plot_scatter(X, y):
    """Scatter plot of distances.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, 2)
        Predictions, distances or probabilities.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.

    Returns
    -------
    fig : matplotlib figure
        Figure of scatter plot.

    Notes
    -----
    .. versionadded:: 0.2.0
    """

    if X.ndim != 2:
        raise ValueError("Input X has not 2 dimensions")
    if X.shape[1] != 2:
        raise ValueError("Input X has not 2 classes")

    fig, ax = plt.subplots(figsize=(7, 7))

    classes = np.unique(y)
    class0 = X[y == classes[0]]
    class1 = X[y == classes[1]]
    ax.scatter(class0[:, 0], class0[:, 1], alpha=1, color="red", label=classes[0])
    ax.scatter(class1[:, 0], class1[:, 1], alpha=0.5, color="blue", label=classes[1])

    ax.legend(title="Classes", loc="upper center")

    ax.plot(ax.get_xlim(), ax.get_ylim(), color="black")
    return fig
