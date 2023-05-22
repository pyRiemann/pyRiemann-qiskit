import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, NaiveQAOAOptimizer
from pyriemann.classification import MDM
from pyriemann.utils.distance import (distance_logeuclid,
                                      logm,
                                      distance_methods,
                                      distance)


def logeucl_dist_convex(X, y, optimizer=NaiveQAOAOptimizer()):
    """Convex formulation of the MDM algorithm
    with log-euclidian metric.
    Parameters
    ----------
    X : ndarray, shape (n_classes, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_channels, n_channels)
        A trial
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
    n_classes, _, _ = X.shape
    classes = range(n_classes)

    def dist(m1, m2): 
        return distance_logeuclid(m1, m2)

    prob = Model()

    # should be part of the optimizer
    w = optimizer.get_weights(prob, classes)

    print(logm(X[0]), logm(X[1]))

    _2VecLogYD = 2 * prob.sum(w[i] * dist(y, X[i]) for i in classes)

    wtDw = prob.sum(w[i] * w[j] * dist(X[i], X[j]) for i in classes for j in classes)

    objectives = wtDw - _2VecLogYD

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob, reshape=False)

    return result
