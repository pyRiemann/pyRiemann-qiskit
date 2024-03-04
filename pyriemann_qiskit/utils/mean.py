from typing_extensions import deprecated
from docplex.mp.model import Model
from pyriemann.utils.mean import mean_functions
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.utils.base import logm, expm
from qiskit_optimization.algorithms import ADMMOptimizer


@deprecated(
    "fro_mean_convex is deprecated and will be removed in 0.3.0; "
    "please use qmean_euclid."
)
def fro_mean_convex():
    pass


def qmean_euclid(X, sample_weight=None, optimizer=ClassicalOptimizer()):
    """Euclidean mean with Constraint Programming Model.

    Constraint Programming Model (CPM) [1]_ formulation of the mean
    with Euclidean distance.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. Never used in practice.
        It is kept only for standardization with pyRiemann.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Returns
    -------
    mean : ndarray, shape (n_channels, n_channels)
        CPM-optimized Euclidean mean.

    Notes
    -----
    .. versionadded:: 0.0.3
    .. versionchanged:: 0.0.4
        Add regularization of the results.
    .. versionchanged:: 0.2.0
        Rename from `fro_mean_convex` to `qmean_euclid`.
        Remove shrinkage.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html
    """

    optimizer = get_global_optimizer(optimizer)

    n_matrices, n_channels, _ = X.shape
    channels = range(n_channels)
    matrices = range(n_matrices)

    prob = Model()

    X_mean = optimizer.covmat_var(prob, channels, "fro_mean")

    def _dist_euclid(A, B):
        A = optimizer.convert_covmat(A)
        return prob.sum_squares(A[r, c] - B[r, c] for r in channels for c in channels)

    objectives = prob.sum(_dist_euclid(X[i], X_mean) for i in matrices)

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob)

    return result


def qmean_logeuclid(
    X, sample_weight=None, optimizer=ClassicalOptimizer(optimizer=ADMMOptimizer())
):
    """Log-Euclidean mean with Constraint Programming Model.

    Constraint Programming Model (CPM) [2]_ formulation of the mean
    with Log-Euclidean distance [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    sample_weights : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. Never used in practice.
        It is kept only for standardization with pyRiemann.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Returns
    -------
    mean : ndarray, shape (n_channels, n_channels)
        CPM-optimized Log-Euclidean mean.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        Geometric means in a novel vector space structure on
        symmetric positive-definite matrices
        V. Arsigny, P. Fillard, X. Pennec, and N. Ayache.
        SIAM Journal on Matrix Analysis and Applications. Volume 29, Issue 1 (2007).
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html
    """

    log_X = logm(X)
    result = qmean_euclid(log_X, sample_weight, optimizer)
    return expm(result)


mean_functions["qeuclid"] = qmean_euclid
mean_functions["qlogeuclid"] = qmean_logeuclid
