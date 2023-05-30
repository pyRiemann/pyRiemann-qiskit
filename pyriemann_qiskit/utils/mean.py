from docplex.mp.model import Model
from pyriemann.utils.mean import mean_methods
from pyriemann_qiskit.utils.docplex import (ClassicalOptimizer,
                                            get_global_optimizer)
from pyriemann.estimation import Shrinkage
from sklearn.covariance import ledoit_wolf


def fro_mean_convex(covmats, sample_weight=None,
                    optimizer=ClassicalOptimizer()):
    """Convex formulation of the mean
    with frobenius distance.
    Parameters
    ----------
    covmats: ndarray, shape (n_classes, n_channels, n_channels)
        Set of SPD matrices.
    sample_weights:  None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. Never used in practice.
        It is kept only for standardization with pyRiemann.
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    mean : ndarray, shape (n_channels, n_channels)
        Convex-optimized forbenius mean.

    Notes
    -----
    .. versionadded:: 0.0.3
    .. versionmodified:: 0.0.4
      Add regularization of the results.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """

    optimizer = get_global_optimizer(optimizer)

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

    # regularize output
    reg_mean_cov = ledoit_wolf(result)
    return reg_mean_cov


mean_methods["convex"] = fro_mean_convex
