"""Riemannian Adam Optimizer for Variational Quantum Circuits.

This module implements Adam [1]_ with a manifold-aware retraction step,
for optimizing variational quantum circuit parameters on Riemannian
manifolds (Bloch sphere).

Standard Adam, as shipped in ``qiskit_algorithms.optimizers.ADAM``, treats
parameters as living in unconstrained Euclidean space: its ``bounds``
support level is ``ignored``, so nothing stops iterates from drifting
outside a bounded interval such as ``[0, pi]``.

Gate-rotation angles instead live on a flat manifold (a product of circles
for full-period parameters, a bounded interval for others). Because that
manifold has zero curvature, the moment estimates need no parallel
transport between tangent spaces — only a retraction step after each
update: wrap angles with a ``2*pi`` period back into range, and clip
angles with a non-periodic bound (e.g. QIOCE's ``[0, pi]``) instead of
letting them escape the valid domain.

Gradients are approximated via central finite differences, so no
analytical gradient is required.

References
----------
.. [1] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
       optimization. arXiv:1412.6980.
.. [2] Becigneul, G., & Ganea, O. E. (2019). Riemannian adaptive
       optimization methods. ICLR.
"""

import numpy as np
from qiskit_algorithms.optimizers import (
    Optimizer,
    OptimizerResult,
    OptimizerSupportLevel,
)


class RiemannianAdamOptimizer(Optimizer):
    """Adam optimizer with manifold-aware retraction for VQC parameters,
    inspired by [1]_.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations.
    lr : float, default=0.1
        Learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimate.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    fd_epsilon : float, default=1e-5
        Finite-difference step size for gradient approximation.
        Should be small (~1e-5 to 1e-7) for accurate numerical gradients.
    tol : float, default=1e-6
        Convergence tolerance on the gradient norm.

    Attributes
    ----------
    trajectory_ : list of ndarray
        Optimization trajectory (parameter history).
    loss_history_ : list of float
        Loss function values at each iteration.

    Examples
    --------
    >>> from pyriemann_qiskit.optimization.riemannian_adam import (
    ...     RiemannianAdamOptimizer
    ... )
    >>> optimizer = RiemannianAdamOptimizer(maxiter=100, lr=0.1)
    >>> # Use with QuanticNCH or other quantum classifiers
    >>> # qaoa_optimizer=optimizer

    Notes
    -----
    .. versionadded:: 0.7.0

    References
    ----------
    .. [1] Becigneul, G., & Ganea, O. E. (2019). Riemannian adaptive
           optimization methods. ICLR.
    """

    def __init__(
        self,
        maxiter=100,
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        fd_epsilon=1e-5,
        tol=1e-6,
    ):
        super().__init__()
        self._maxiter = maxiter
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._fd_epsilon = fd_epsilon
        self._tol = tol
        self.trajectory_ = []
        self.loss_history_ = []

    def minimize(self, fun, x0, jac=None, bounds=None):
        """Minimize the objective function using Riemannian Adam.

        Parameters
        ----------
        fun : callable
            Objective function to minimize. Should accept a 1D array
            and return a scalar.
        x0 : ndarray
            Initial parameter vector.
        jac : callable, optional
            Gradient function (not used, included for compatibility).
        bounds : list of tuples, optional
            Parameter bounds. A bound spanning exactly ``2*pi`` is
            treated as periodic (wrapped); any other bound is clipped.

        Returns
        -------
        OptimizerResult
            Result object containing:

            - x: optimal parameters
            - fun: objective value at optimal parameters
            - nfev: number of function evaluations
            - nit: number of iterations
        """
        x = np.array(x0, dtype=float)
        n = len(x)
        nfev = 0

        f_current = fun(x)
        nfev += 1

        self.trajectory_ = [x.copy()]
        self.loss_history_ = [f_current]

        # Precompute vectorised bounds arrays once (avoids repeated per-element
        # work). Only treat as periodic if the span is a full 2*pi rotation
        # period. Bounds like [0, pi] (QIOCE) must be clipped, not wrapped —
        # wrapping with period pi teleports parameters to the wrong end of
        # the range.
        if bounds is not None:
            lower_b = np.array([b[0] if b[0] is not None else np.nan for b in bounds])
            upper_b = np.array([b[1] if b[1] is not None else np.nan for b in bounds])
            span = upper_b - lower_b
            periodic = ~(np.isnan(lower_b) | np.isnan(upper_b)) & np.isclose(
                span, 2 * np.pi
            )
            period_b = np.where(periodic, span, 1.0)  # dummy for non-periodic
            has_lower = ~np.isnan(lower_b) & ~periodic
            has_upper = ~np.isnan(upper_b) & ~periodic

        m = np.zeros(n)
        v = np.zeros(n)

        iteration = 0
        for iteration in range(self._maxiter):
            # Estimate gradient using central differences.
            # Modify x[i] in-place and restore — avoids 2n array allocations
            # per iter. Clamp perturbations to stay within bounds so loss is
            # never evaluated at an illegal parameter value.
            grad = np.zeros(n)
            for i in range(n):
                orig = x[i]
                step = self._fd_epsilon

                x_plus = orig + step
                x_minus = orig - step
                if bounds is not None:
                    lo, hi = bounds[i]
                    if lo is not None:
                        x_minus = max(x_minus, lo)
                    if hi is not None:
                        x_plus = min(x_plus, hi)
                actual_step = (x_plus - x_minus) / 2.0

                x[i] = x_plus
                f_plus = fun(x)
                nfev += 1
                x[i] = x_minus
                f_minus = fun(x)
                nfev += 1
                x[i] = orig
                if actual_step != 0.0:
                    grad[i] = (f_plus - f_minus) / (2.0 * actual_step)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < self._tol:
                break

            # Adam moment updates (Euclidean — the manifold here is flat,
            # so no parallel transport is needed between iterates).
            t = iteration + 1
            m = self._beta1 * m + (1 - self._beta1) * grad
            v = self._beta2 * v + (1 - self._beta2) * grad**2
            m_hat = m / (1 - self._beta1**t)
            v_hat = v / (1 - self._beta2**t)

            x_new = x - self._lr * m_hat / (np.sqrt(v_hat) + self._eps)

            # Riemannian retraction back onto the parameter manifold —
            # vectorised, same convention as AndersonAccelerationOptimizer.
            if bounds is not None:
                x_new = np.where(
                    periodic,
                    lower_b + np.mod(x_new - lower_b, period_b),
                    x_new,
                )
                x_new = np.where(has_lower, np.maximum(x_new, lower_b), x_new)
                x_new = np.where(has_upper, np.minimum(x_new, upper_b), x_new)

            x = x_new
            f_current = fun(x)
            nfev += 1

            self.trajectory_.append(x.copy())
            self.loss_history_.append(f_current)

        result = OptimizerResult()
        result.x = x
        result.fun = f_current
        result.nfev = nfev
        result.nit = iteration + 1 if iteration >= 0 else 0
        return result

    @property
    def settings(self):
        """Return optimizer settings."""
        return {
            "maxiter": self._maxiter,
            "lr": self._lr,
            "beta1": self._beta1,
            "beta2": self._beta2,
            "eps": self._eps,
            "fd_epsilon": self._fd_epsilon,
            "tol": self._tol,
        }

    def get_support_level(self):
        """Return support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }
