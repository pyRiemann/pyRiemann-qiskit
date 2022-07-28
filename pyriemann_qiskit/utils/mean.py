import numpy as np
from docplex.mp.model import Model
from pyriemann.utils.mean import mean_methods
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer


def fro_mean_convex(covmats, sample_weight=None,
                    optimizer=ClassicalOptimizer()):
    n_trials, n_channels, _ = covmats.shape
    channels = range(n_channels)
    trials = range(n_trials)

    prob = Model()

    X_mean = optimizer.covmat_var(prob, channels, 'fro_mean')

    def _fro_dist(A, B):
        A = optimizer.convert_covmat(A)
        return prob.sum_squares(A[r, c] - B[r, c]
                                for r in channels
                                for c in channels)

    objectives = prob.sum(_fro_dist(covmats[i], X_mean) for i in trials)

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob)

    return result


mean_methods["convex"] = fro_mean_convex
