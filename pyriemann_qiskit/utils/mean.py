from docplex.mp.model import Model
from pyriemann.utils.mean import mean_functions
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.estimation import Shrinkage


def fro_mean_convex(
    covmats, sample_weight=None, optimizer=ClassicalOptimizer(), shrink=True
):
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
    shrink: boolean (default: true)
        If True, it applies shrinkage regularization [2]_
        of the resulting covariance matrix.

    Returns
    -------
    mean : ndarray, shape (n_channels, n_channels)
        Convex-optimized forbenius mean.

    Notes
    -----
    .. versionadded:: 0.0.3
    .. versionchanged:: 0.0.4
        Add regularization of the results.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    .. [2] \
        https://pyriemann.readthedocs.io/en/v0.4/generated/pyriemann.estimation.Shrinkage.html
    """

    optimizer = get_global_optimizer(optimizer)

    n_trials, n_channels, _ = covmats.shape
    channels = range(n_channels)
    trials = range(n_trials)

    prob = Model()

    X_mean = optimizer.covmat_var(prob, channels, "fro_mean")

    def _fro_dist(A, B):
        A = optimizer.convert_covmat(A)
        return prob.sum_squares(A[r, c] - B[r, c] for r in channels for c in channels)

    objectives = prob.sum(_fro_dist(covmats[i], X_mean) for i in trials)

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob)

    if shrink:
        return Shrinkage(shrinkage=0.9).transform([result])[0]
    return result


mean_functions["convex"] = fro_mean_convex
