"""Quantum distance metrics for SPD matrices.

Notes
-----
.. versionchanged:: 0.6.0
    Moved from ``pyriemann_qiskit.utils.distance`` to
    ``pyriemann_qiskit.optimization.distance``.
"""

from copy import deepcopy

import numpy as np
from docplex.mp.model import Model
from pyriemann.utils.base import logm
from pyriemann.utils.distance import (
    distance_euclid,
    distance_functions,
    distance_logeuclid,
)
from pyriemann.utils.mean import mean_logeuclid

from .docplex import ClassicalOptimizer


def qdistance_logeuclid_to_convex_hull(
    A, B, optimizer=ClassicalOptimizer(), resolution_bounds=None
):
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
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of
        :class:`pyriemann_qiskit.optimization.docplex.pyQiskitOptimizer`.
    resolution_bounds : sequence of int or None, default=None
        Integer upper bounds to evaluate for an integer encoded optimizer.
        The configured bound must be included. The result uses the stage with
        the smallest original Log-Euclidean distance.

    Returns
    -------
    distance : float
        Log-Euclidean distance between the SPD matrix B and the convex hull of
        the set of SPD matrices A, defined as the distance between B and the
        matrix of the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        'Convex Class Model on Symmetric Positive Definite Manifolds',
        Image and Vision Computing, 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    weights = weights_logeuclid_to_convex_hull(
        A, B, optimizer, resolution_bounds=resolution_bounds
    )
    # compute nearest matrix
    C = mean_logeuclid(A, weights)
    distance = distance_logeuclid(C, B)

    return distance


def weights_logeuclid_to_convex_hull(
    A, B, optimizer=ClassicalOptimizer(), resolution_bounds=None
):
    """Weights for Log-Euclidean distance to a convex hull of SPD matrices.

    Weights for Log-Euclidean distance between a SPD matrix B
    and the convex hull of a set of SPD matrices A [1]_,
    formulated as a Constraint Programming Model (CPM) [2]_.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of
        :class:`pyriemann_qiskit.optimization.docplex.pyQiskitOptimizer`.
    resolution_bounds : sequence of int or None, default=None
        Integer upper bounds to evaluate for an integer encoded optimizer.
        The configured bound must be included. The returned weights minimize
        the original Log-Euclidean distance among the evaluated resolutions.


    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        Optimized weights for the set of SPD matrices A.
        Using these weights, the weighted Log-Euclidean mean of set A
        provides the matrix of the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.2.0
        Rename from `logeucl_dist_convex` to `weights_logeuclid_to_convex_hull`.
        Add linear constraint on weights (sum = 1).

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        'Convex Class Model on Symmetric Positive Definite Manifolds',
        Image and Vision Computing, 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    n_matrices, _, _ = A.shape
    matrices = range(n_matrices)

    def trace_prod_log(m1, m2):
        return np.trace(logm(m1) @ logm(m2))

    configured_bound, bounds = _get_bounds(optimizer, resolution_bounds)

    best_weights = None
    best_distance = np.inf
    evaluation_history = []
    for bound in bounds:
        stage_optimizer = optimizer
        if bound != configured_bound:
            stage_optimizer = deepcopy(optimizer)
            stage_optimizer.upper_bound = bound

        prob = Model()
        raw_weights = stage_optimizer.get_weights(prob, matrices)
        if bound is None:
            w = raw_weights
        else:
            if bound <= 0:
                raise ValueError("integer encoding upper_bound must be positive")
            # Integer weights are represented on a grid. Impose the simplex
            # at that grid's scale and use normalized weights in the objective.
            w = raw_weights / bound

        wt_log_a_log_a = prob.sum(
            w[i] * w[j] * trace_prod_log(A[i], A[j]) for i in matrices for j in matrices
        )
        w_log_b_log_a = prob.sum(w[i] * trace_prod_log(B, A[i]) for i in matrices)
        prob.set_objective("min", wt_log_a_log_a - 2 * w_log_b_log_a)
        if bound is None:
            prob.add_constraint(prob.sum(w) == 1)
        else:
            prob.add_constraint(prob.sum(raw_weights) == bound)

        weights = stage_optimizer.solve(prob, reshape=False)
        weights = stage_optimizer._decode_hull_weights(weights, bound)
        if bound is not None:
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < -1e-8)
                or np.any(weights > 1 + 1e-8)
                or not np.isclose(np.sum(weights), 1.0, atol=1e-8)
            ):
                evaluation_history.extend(
                    getattr(
                        stage_optimizer,
                        "evaluated_values_",
                        getattr(stage_optimizer, "y_", []),
                    )
                )
                continue
        distance = distance_logeuclid(mean_logeuclid(A, weights), B)
        if distance < best_distance:
            best_weights = weights
            best_distance = distance
        evaluation_history.extend(
            getattr(
                stage_optimizer,
                "evaluated_values_",
                getattr(stage_optimizer, "y_", []),
            )
        )

    if resolution_bounds is not None:
        if hasattr(optimizer, "evaluated_values_"):
            optimizer.evaluated_values_ = evaluation_history
        if hasattr(optimizer, "y_"):
            optimizer.y_ = evaluation_history
    if best_weights is None:
        raise RuntimeError("integer optimizer did not return feasible simplex weights")
    return best_weights


def _get_bounds(optimizer, resolution_bounds):
    """Return the configured integer bound and resolution schedule."""
    configured_bound = getattr(optimizer.encoding, "upper_bound", None)
    if configured_bound is not None and configured_bound <= 0:
        raise ValueError("integer encoding upper_bound must be positive")
    if resolution_bounds is None:
        return configured_bound, [configured_bound]
    if configured_bound is None:
        raise ValueError("resolution_bounds requires an integer encoded optimizer")

    bounds = list(resolution_bounds)
    if (
        not bounds
        or configured_bound not in bounds
        or any(
            not isinstance(bound, (int, np.integer))
            or isinstance(bound, (bool, np.bool_))
            or bound <= 0
            for bound in bounds
        )
        or len(set(bounds)) != len(bounds)
    ):
        raise ValueError(
            "resolution_bounds must contain distinct positive integers "
            "including the optimizer's configured upper_bound"
        )
    return configured_bound, bounds


def _weights_distance(
    A, B, distance=distance_logeuclid, optimizer=ClassicalOptimizer()
):
    """`distance` weights between a SPD and a set of SPD matrices.

    `distance` weights between a SPD matrix B and each SPD matrix inside A,
    formulated as a Constraint Programming Model (CPM) [1]_.
    The higher weight corresponds to the closer SPD matrix inside A,
    which is closer to B.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    distance : Callable[[ndarray, ndarray], float]
        One of the pyRiemann distance.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of
        :class:`pyriemann_qiskit.optimization.docplex.pyQiskitOptimizer`.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        Optimized weights for the set of SPD matrices A.
        The higher weight corresponds to the closer SPD matrix inside A,
        which is closer to B.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    n_matrices, _, _ = A.shape
    matrices = range(n_matrices)

    prob = Model()

    w = optimizer.get_weights(prob, matrices)

    objectif = prob.sum(w[i] * distance(B, A[i]) for i in matrices)

    prob.set_objective("min", objectif)
    prob.add_constraint(prob.sum(w) == 1)
    weights = optimizer.solve(prob, reshape=False)

    return weights


distance_functions["qlogeuclid_hull"] = weights_logeuclid_to_convex_hull
distance_functions["qeuclid"] = lambda A, B, optimizer: _weights_distance(
    A, B, distance_euclid, optimizer
)
distance_functions["qlogeuclid"] = lambda A, B, optimizer: _weights_distance(
    A, B, distance_logeuclid, optimizer
)
