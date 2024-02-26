import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.classification import MDM
from pyriemann.utils.distance import distance_functions, distance_logeuclid
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_logeuclid
from typing_extensions import deprecated


@deprecated(
    "logeucl_dist_convex is deprecated and will be removed in 0.3.0; "
    "please use distance_logeuclid_cpm."
)
def logeucl_dist_convex():
    pass


def distance_logeuclid_cpm(A, B, optimizer=ClassicalOptimizer(), return_weights=False):
    """Log-Euclidean distance to a convex hull of SPD matrices.

    Log-Euclidean distance between a SPD matrix B and the convex hull of a set
    of SPD matrices A [1]_, formulated as a Constraint Programming Model (CPM)
    [2]_.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        The optimized weights for the set of SPD matrices A.
        Using these weights, the weighted Log-Euclidean mean of set A
        provides the matrix of the convex hull closest to matrix B.
    distance : float
        Log-Euclidean distance between the SPD matrix B and the convex hull of the set
        of SPD matrices A, defined as the distance between B and the matrix of
        the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.0.4

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        ‘Convex Class Model on Symmetric Positive Definite Manifolds’,
        Image and Vision Computing, 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """

    optimizer = get_global_optimizer(optimizer)

    n_matrices, _, _ = A.shape
    matrices = range(n_matrices)

    def log_prod(m1, m2):
        return np.nansum(logm(m1).flatten() * logm(m2).flatten())

    prob = Model()

    # should be part of the optimizer
    w = optimizer.get_weights(prob, matrices)

    _2VecLogYD = 2 * prob.sum(w[i] * log_prod(B, A[i]) for i in matrices)

    wtDw = prob.sum(
        w[i] * w[j] * log_prod(A[i], A[j]) for i in matrices for j in matrices
    )

    objectives = wtDw - _2VecLogYD

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob, reshape=False)

    # compute nearest matrix and distance
    C = mean_logeuclid(A, result)
    distance = distance_logeuclid(C, B)

    return 1 - result, distance


_mdm_predict_distances_original = MDM._predict_distances


def predict_distances(mdm, X):
    if mdm.metric_dist == "logeuclid_cpm":
        centroids = np.array(mdm.covmeans_)
        return np.array([distance_logeuclid_cpm(centroids, x)[0] for x in X])
    else:
        return _mdm_predict_distances_original(mdm, X)


def is_cpm_dist(string):
    """Indicates if the distance is a CPM distance.

    Return True is "string" represents a Constraint Programming Model (CPM) [1]_
    distance available in the library.

    Parameters
    ----------
    string: str
        A string representation of the distance.

    Returns
    -------
    is_cpm_dist : boolean
        True if "string" represents a CPM distance available in the library.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    return "_cpm" in string and string in distance_functions


MDM._predict_distances = predict_distances

# This is only for validation inside the MDM.
# In fact, we override the _predict_distances method
# inside MDM to directly use distance_logeuclid_cpm when the metric is "logeuclid_cpm"
# This is due to the fact the the signature of this method is different from
# the usual distance functions.
distance_functions["logeuclid_cpm"] = distance_logeuclid_cpm
