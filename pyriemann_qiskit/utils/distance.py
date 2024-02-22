import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.classification import MDM
from pyriemann.utils.distance import distance_functions
from pyriemann.utils.base import logm
from typing_extensions import deprecated


@deprecated(
    "logeucl_dist_convex is deprecated and will be removed in 0.3.0; "
    "please use distance_logeuclid_cpm."
)
def logeucl_dist_convex():
    pass


def distance_logeuclid_cpm(A, B, optimizer=ClassicalOptimizer()):
    """Log-Euclidean distance by Constraint Programming Model.

    Constraint Programming Model (CPM) [2]_ formulation of
    the Log-Euclidean distance [1]_.

    Parameters
    ----------
    A : ndarray, shape (n_classes, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        A trial
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    weights : ndarray, shape (n_classes,)
        The weights associated with each class.
        Higher the weight, closer it is to the class prototype.
        Weights are not normalized.

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

    n_classes, _, _ = A.shape
    classes = range(n_classes)

    def log_prod(m1, m2):
        return np.nansum(logm(m1).flatten() * logm(m2).flatten())

    prob = Model()

    # should be part of the optimizer
    w = optimizer.get_weights(prob, classes)

    _2VecLogYD = 2 * prob.sum(w[i] * log_prod(B, A[i]) for i in classes)

    wtDw = prob.sum(
        w[i] * w[j] * log_prod(A[i], A[j]) for i in classes for j in classes
    )

    objectives = wtDw - _2VecLogYD

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob, reshape=False)

    return 1 - result


_mdm_predict_distances_original = MDM._predict_distances


def predict_distances(mdm, X):
    if mdm.metric_dist == "logeuclid_cpm":
        centroids = np.array(mdm.covmeans_)
        return np.array([distance_logeuclid_cpm(centroids, x) for x in X])
    else:
        return _mdm_predict_distances_original(mdm, X)


def is_cpm_dist(string):
    """Return True is "string" represents a Constraint Programming Model (CPM) [1]_
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
