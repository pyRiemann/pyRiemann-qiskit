import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm


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


def cone_surface(radius, height, num_points=100):
    def f(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:80j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = f(x, y)

    return x, y, z

def to_xyz(points):
        return np.array([[point[0, 0], point[0, 1], point[1, 1]] for point in points])

# classif: cov 2d -> cart -> PCA -> MDM?
def plot_cone(points, y_test):
    def to_xyz(points):
        return np.array([[point[0, 0], point[0, 1], point[1, 1]] for point in points])

    cart = to_xyz(points[y_test == 'Target', :])
    cart2 = to_xyz(points[y_test == 'NonTarget', :])
    cart0 = to_xyz(points)
    pca = PCA().fit(cart0).components_
    x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
    # cart = cart @ pca
    # cart2 = cart2 @ pca
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], color='black')
    ax.scatter(cart2[:, 0], cart2[:, 1], cart2[:, 2], alpha=0.5, color='blue')
    ax.scatter(cart[:, 0], cart[:, 1], cart[:, 2], alpha=0.5)

    from scipy.spatial import ConvexHull
    hull = ConvexHull(cart0)
    for simplex in hull.simplices:
        plt.plot(cart0[simplex, 0], cart0[simplex, 1], cart0[simplex, 2], 'k--', alpha=0.2)

    
    # X, Y, Z = cone_surface(1, 3)
    # print(X.shape, Y.shape, Z.shape)
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    # ax.plot_surface(X, Y, -Z, alpha=0.5)
    plt.show()

import scipy
def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull