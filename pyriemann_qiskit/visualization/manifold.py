import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm
from pyriemann_qiskit.utils.math import to_xyz

def plot_manifold(X, y):

    classes = np.unique(y)

    points = to_xyz(X)
    points0 = points[y == classes[0]]
    points1 = points[y == classes[1]]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], alpha=1, color='blue')
    ax.scatter(points0[:, 0], points0[:, 1], points0[:, 2], alpha=0.5)

    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k--', alpha=0.2)

    plt.show()
