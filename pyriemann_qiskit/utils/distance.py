import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import (ClassicalOptimizer,
                                            get_global_optimizer)
from pyriemann.classification import MDM
from pyriemann.utils.distance import distance_methods
from pyriemann.utils.base import logm


def logeucl_dist_convex(X, y, optimizer=ClassicalOptimizer()):
    """Convex formulation of the MDM algorithm
    with log-euclidian metric.

    Parameters
    ----------
    X : ndarray, shape (n_classes, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_channels, n_channels)
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
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """

    optimizer = get_global_optimizer(optimizer)

    n_classes, _, _ = X.shape
    classes = range(n_classes)

    def dist(m1, m2):
        return np.nansum(logm(m1).flatten() * logm(m2).flatten())

    prob = Model()

    # should be part of the optimizer
    w = optimizer.get_weights(prob, classes)

    _2VecLogYD = 2 * prob.sum(w[i] * dist(y, X[i]) for i in classes)

    wtDw = prob.sum(w[i] * w[j] * dist(X[i], X[j]) for i in classes
                    for j in classes)

    objectives = wtDw - _2VecLogYD

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob, reshape=False)

    return 1 - result


_mdm_predict_distances_original = MDM._predict_distances


def predict_distances(mdm, X):
    if mdm.metric_dist == 'convex':
        centroids = np.array(mdm.covmeans_)
        return np.array([logeucl_dist_convex(centroids, x) for x in X])
    else:
        return _mdm_predict_distances_original(mdm, X)


MDM._predict_distances = predict_distances

# This is only for validation inside the MDM.
# In fact, we override the _predict_distances method
# inside MDM to directly use logeucl_dist_convex when the metric is "convex"
# This is due to the fact the the signature of this method is different from
# the usual distance functions.
distance_methods['convex'] = logeucl_dist_convex
