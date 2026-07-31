"""COBYLA optimizer with bounds support, for docplex-based constrained problems.

Notes
-----
.. versionadded:: 0.7.0
    Extracted from :class:`pyriemann_qiskit.optimization.docplex.ClassicalOptimizer`
    as part of the migration away from the archived ``qiskit-optimization``
    package. Its ``qiskit_optimization.algorithms.CobylaOptimizer`` handled
    variable bounds — which scipy.optimize.minimize's COBYLA method does not
    support natively — by translating them into inequality constraints; this
    module reproduces that workaround so callers can treat `bounds` as if it
    were natively supported.
"""

from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, OptimizerSupportLevel
from qiskit_addon_opt_mapper import INFINITY
from scipy.optimize import minimize as scipy_minimize


def _bounds_as_constraints(bounds):
    """Workaround for COBYLA's lack of native bounds support.

    scipy.optimize.minimize's COBYLA method ignores the `bounds` argument, so
    bounds must instead be expressed as inequality constraints.

    Parameters
    ----------
    bounds : list of tuple(float | None, float | None)
        The (lowerbound, upperbound) of each decision variable.

    Returns
    -------
    constraints : list of dict
        scipy.optimize.minimize-style inequality constraints.
    """
    constraints = []
    for i, (lb, ub) in enumerate(bounds):
        if lb is not None and lb > -INFINITY:
            constraints.append({"type": "ineq", "fun": lambda x, lb=lb, i=i: x[i] - lb})
        if ub is not None and ub < INFINITY:
            constraints.append({"type": "ineq", "fun": lambda x, ub=ub, i=i: ub - x[i]})
    return constraints


class CobylaOptimizer(Optimizer):
    """COBYLA optimizer with bounds support.

    Wraps ``scipy.optimize.minimize`` (method="COBYLA"), translating `bounds`
    into inequality constraints since COBYLA does not support `bounds`
    natively. Additional constraints (e.g. derived from the linear
    constraints of a docplex model) can be supplied via `constraints`.

    Parameters
    ----------
    rhobeg : float, default=2.1
        Reasonable initial changes to the variables.
    tol : float, default=0.000001
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    maxiter : int, default=1000
        Maximum number of function evaluations.
    disp : bool, default=False
        Set to True to print convergence messages.
    constraints : list of dict | None, default=None
        Additional scipy.optimize.minimize-style constraints.

    Notes
    -----
    .. versionadded:: 0.7.0
    """

    def __init__(self, rhobeg=2.1, tol=0.000001, maxiter=1000, disp=False, constraints=None):
        super().__init__()
        self.rhobeg = rhobeg
        self.tol = tol
        self.maxiter = maxiter
        self.disp = disp
        self.constraints = constraints if constraints is not None else []

    def minimize(self, fun, x0, jac=None, bounds=None):
        constraints = list(self.constraints)
        if bounds:
            constraints += _bounds_as_constraints(bounds)

        raw_result = scipy_minimize(
            fun=fun,
            x0=x0,
            method="COBYLA",
            constraints=constraints,
            options={"rhobeg": self.rhobeg, "maxiter": self.maxiter, "disp": self.disp},
            tol=self.tol,
        )

        result = OptimizerResult()
        result.x = raw_result.x
        result.fun = raw_result.fun
        result.nfev = raw_result.nfev
        result.nit = raw_result.get("nit", None)
        return result

    @property
    def settings(self):
        """Return optimizer settings."""
        return {
            "rhobeg": self.rhobeg,
            "tol": self.tol,
            "maxiter": self.maxiter,
            "disp": self.disp,
            "constraints": self.constraints,
        }

    def get_support_level(self):
        """Return support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }
