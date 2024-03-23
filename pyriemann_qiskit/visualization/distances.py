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
    print(X[y == classes[0], 0])
    print(1 - X[y == classes[1], 1])
    ax.hist(X[y == classes[0], 0], label=classes[0], alpha=0.5)
    ax.hist(1 - X[y == classes[1], 1], label=classes[1], alpha=0.5)
    ax.set(xlabel="Distances", ylabel="Frequency")
    ax.legend(title="Classes", loc="upper center")

    return fig

def plot_scatter(X, y):
    classes = np.unique(y)
    class0 = X[y == classes[0], :]
    class1 = X[y == classes[1], :]
    sns.scatterplot(x=class0[:, 0], y=class0[:, 1], alpha=1, color='red')
    sns.scatterplot(x=class1[:, 0], y=class1[:, 1], alpha=0.4, color='blue')
    max_point_x = max(class0[:, 0].max(), class1[:, 0].max())
    max_point_y = max(class0[:, 1].max(), class1[:, 1].max())
    sns.lineplot(x=[0, max_point_x], y=[0, max_point_y], color='black')
    plt.show()