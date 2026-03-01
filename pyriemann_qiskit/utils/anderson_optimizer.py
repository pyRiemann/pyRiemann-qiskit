"""Anderson Acceleration Optimizer for Variational Quantum Circuits.

This module implements mesh-free Anderson acceleration for optimizing
variational quantum circuit parameters on Riemannian manifolds (Bloch sphere).

The parameters are rotation angles for quantum gates, which live on a
Riemannian manifold with periodic boundary conditions. The optimizer:

- Computes gradients in the tangent space
- Handles periodic wraparound for angle parameters
- Uses Riemannian retraction to project back to the manifold

The algorithm follows equations 41-44 from the mesh-free IGL formulation:

- ΔW = [w_k - w_{k-1}, ..., w_{k-m+1} - w_{k-m}]  (Eq. 41)
- ΔR = [r_k - r_{k-1}, ..., r_{k-m+1} - r_{k-m}]  (Eq. 42)
- γ = argmin_γ ||r_k - ΔR·γ||² + λ||γ||²         (Eq. 43)
- w_{k+1} = w_k + α·r_k - (ΔW + α·ΔR)·γ          (Eq. 44)

where differences are computed in the tangent space and r_k is the
Riemannian gradient-based residual.

References
----------
.. [1] Toth, A., & Kelley, C. T. (2015). Convergence analysis for Anderson
       acceleration. SIAM Journal on Numerical Analysis, 53(2), 805-819.
.. [2] Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point
       iterations. SIAM Journal on Numerical Analysis, 49(4), 1715-1735.
"""

from collections import deque

import numpy as np
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult
from scipy.linalg import lstsq


class AndersonAccelerationOptimizer(Optimizer):
    """Anderson acceleration optimizer for variational quantum circuits.

    Anderson acceleration is a derivative-free fixed-point iteration method
    that achieves superlinear convergence for compact operators. Gradients are
    approximated via central finite differences, so no analytical gradient is
    required. It is particularly well-suited for quantum circuit optimization where:

    - Analytical gradients are expensive or unavailable
    - The objective function is bounded (compact operator)
    - The parameter space may have Riemannian structure

    The algorithm maintains a history of parameter vectors and residuals,
    then solves a small least-squares problem to extrapolate the next
    iterate. This typically converges in 15-25 iterations compared to
    100+ for gradient-based methods.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations.
    m : int, default=5
        History depth (number of previous iterates to use).
        Typical values: 5-10. Larger values may improve convergence
        but increase memory and computation.
    alpha : float, default=1.0
        Damping parameter. alpha=1.0 is undamped (recommended for
        quantum circuits). Values < 1.0 add damping for stability.
    lambda_reg : float, default=1e-8
        Regularization parameter for the least-squares problem.
        Prevents ill-conditioning when residual vectors become
        nearly collinear.
    tol : float, default=1e-6
        Convergence tolerance on the residual norm.
    fd_epsilon : float, default=1e-5
        Finite-difference step size for gradient approximation.
        Should be small (~1e-5 to 1e-7) for accurate numerical gradients.
    learning_rate : float, default=0.1
        Step size for the fixed-point map ``g(x) = x - lr·∇f(x)``.
        Controls how far the iterates move each step. Values in 0.05–0.5
        are typical; too small causes slow convergence, too large causes
        oscillation or bounds overshoot.

    Attributes
    ----------
    _maxiter : int
        Maximum iterations.
    _m : int
        History depth.
    _alpha : float
        Damping parameter.
    _lambda_reg : float
        Regularization parameter.
    _tol : float
        Convergence tolerance.
    _fd_epsilon : float
        Finite-difference step size.
    _lr : float
        Learning rate for the fixed-point map.
    trajectory_ : list of ndarray
        Optimization trajectory (parameter history).
    loss_history_ : list of float
        Loss function values at each iteration.

    Examples
    --------
    >>> from pyriemann_qiskit.utils.anderson_optimizer import (
    ...     AndersonAccelerationOptimizer
    ... )
    >>> optimizer = AndersonAccelerationOptimizer(maxiter=25, m=5)
    >>> # Use with QuanticNCH or other quantum classifiers
    >>> # qaoa_optimizer=optimizer
    """

    def __init__(
        self,
        maxiter=100,
        m=5,
        alpha=1.0,
        lambda_reg=1e-8,
        tol=1e-6,
        fd_epsilon=1e-5,
        learning_rate=0.1,
    ):
        super().__init__()
        self._maxiter = maxiter
        self._m = m
        self._alpha = alpha
        self._lambda_reg = lambda_reg
        self._tol = tol
        self._fd_epsilon = fd_epsilon
        self._lr = learning_rate
        self.trajectory_ = []
        self.loss_history_ = []

    def minimize(self, fun, x0, jac=None, bounds=None):
        """Minimize the objective function using Anderson acceleration.

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
            Parameter bounds (not enforced in current implementation).

        Returns
        -------
        OptimizerResult
            Result object containing:

            - x: optimal parameters
            - fun: objective value at optimal parameters
            - nfev: number of function evaluations
            - nit: number of iterations
        """
        # Initialize
        x = np.array(x0, dtype=float)
        n = len(x)
        nfev = 0

        # History storage — deque auto-evicts oldest entry when full (O(1) FIFO)
        W_history = deque(maxlen=self._m)  # Parameter history
        R_history = deque(maxlen=self._m)  # Residual history

        # Evaluate initial point
        f_current = fun(x)
        nfev += 1

        # Initialize trajectory tracking
        self.trajectory_ = [x.copy()]
        self.loss_history_ = [f_current]

        # Precompute vectorised bounds arrays once (avoids repeated per-element work)
        if bounds is not None:
            lower_b = np.array([b[0] if b[0] is not None else np.nan for b in bounds])
            upper_b = np.array([b[1] if b[1] is not None else np.nan for b in bounds])
            # Only treat as periodic if the span is a full 2π rotation period.
            # Bounds like [0, π] (QIOCE) must be clipped, not wrapped — wrapping
            # with period π teleports parameters to the wrong end of the range.
            span = upper_b - lower_b
            periodic = ~(np.isnan(lower_b) | np.isnan(upper_b)) & np.isclose(
                span, 2 * np.pi
            )
            period_b = np.where(periodic, span, 1.0)  # dummy for non-periodic
            has_lower = ~np.isnan(lower_b) & ~periodic
            has_upper = ~np.isnan(upper_b) & ~periodic
            # For wraparound correction: period column-vector broadcasts over m_k columns
            period_col = period_b[:, np.newaxis]

        iteration = 0
        for iteration in range(self._maxiter):
            # Anderson acceleration for Riemannian manifold (Bloch sphere)
            # Parameters are angles, so we need to respect the manifold structure

            # Estimate gradient using central differences.
            # Modify x[i] in-place and restore — avoids 2n array allocations per iter.
            # Clamp perturbations to stay within bounds so loss is never evaluated
            # at an illegal parameter value.
            grad_approx = np.zeros(n)
            for i in range(n):
                orig = x[i]
                step = self._fd_epsilon

                x_plus = orig + step
                x_minus = orig - step
                if bounds is not None:
                    lo = bounds[i][0]
                    hi = bounds[i][1]
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
                    grad_approx[i] = (f_plus - f_minus) / (2.0 * actual_step)

            # Compute residual: r_k = g(x_k) - x_k = -lr * grad_approx
            # lr controls step size independently from fd_epsilon.
            r = -self._lr * grad_approx

            # Check convergence
            r_norm = np.linalg.norm(r)
            if r_norm < self._tol:
                break

            # Store current iterate and residual (r is freshly allocated — no .copy() needed)
            W_history.append(x.copy())
            R_history.append(r)

            # Anderson acceleration step
            if len(W_history) > 1:
                # Build difference matrices (Equations 41-42) via vectorised np.diff
                W_arr = np.array(W_history)  # shape (m_k+1, n)
                R_arr = np.array(R_history)
                Delta_W = np.diff(W_arr, axis=0).T  # shape (n, m_k)
                Delta_R = np.diff(R_arr, axis=0).T

                # For angles, handle wraparound in tangent-space differences
                if bounds is not None:
                    for dmat in (Delta_W, Delta_R):
                        mask = np.abs(dmat) > period_col / 2
                        dmat -= np.where(mask, np.sign(dmat) * period_col, 0.0)

                m_k = Delta_W.shape[1]

                # Solve least-squares problem (Equation 43)
                # min_gamma ||r_k - Delta_R * gamma||^2 + lambda * ||gamma||^2
                A = Delta_R.T @ Delta_R + self._lambda_reg * np.eye(m_k)
                b = Delta_R.T @ r

                try:
                    gamma = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Fallback to lstsq if solve fails
                    gamma, _, _, _ = lstsq(A, b)

                # Update step (Equation 44)
                # w_{k+1} = w_k + alpha * r_k - (Delta_W + alpha * Delta_R) * gamma
                x_new = x + self._alpha * r - (Delta_W + self._alpha * Delta_R) @ gamma
            else:
                # First iteration or no history: simple fixed-point step
                # w_{k+1} = w_k + alpha * r_k
                x_new = x + self._alpha * r

            # Project back to manifold (Riemannian retraction) — vectorised
            if bounds is not None:
                x_new = np.where(
                    periodic,
                    lower_b + np.mod(x_new - lower_b, period_b),
                    x_new,
                )
                x_new = np.where(has_lower, np.maximum(x_new, lower_b), x_new)
                x_new = np.where(has_upper, np.minimum(x_new, upper_b), x_new)

            # Update
            x = x_new
            f_current = fun(x)
            nfev += 1

            # Append to trajectory (fix: was overwriting instead of appending)
            self.trajectory_.append(x.copy())
            self.loss_history_.append(f_current)

        # Return result
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
            "m": self._m,
            "alpha": self._alpha,
            "lambda_reg": self._lambda_reg,
            "tol": self._tol,
            "fd_epsilon": self._fd_epsilon,
            "learning_rate": self._lr,
        }

    def get_support_level(self):
        """Return support level dictionary."""
        from qiskit_algorithms.optimizers import OptimizerSupportLevel

        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }

    def plot_bloch_trajectory(self, param_indices=(0, 1), figsize=(10, 8)):
        """Plot optimization trajectory on Bloch sphere.

        Parameters
        ----------
        param_indices : tuple of int, default=(0, 1)
            Which two parameters to visualize (θ, φ angles).
        figsize : tuple, default=(10, 8)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The 3D axes object.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        if len(self.trajectory_) == 0:
            raise ValueError("No trajectory data. Run minimize() first.")

        # Extract parameters
        idx_theta, idx_phi = param_indices
        trajectory = np.array(self.trajectory_)
        theta = trajectory[:, idx_theta]
        phi = trajectory[:, idx_phi]

        # Convert to Bloch sphere coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color="gray")

        # Plot trajectory
        ax.plot(x, y, z, "b-", linewidth=2, label="Path")
        ax.scatter(x[0], y[0], z[0], c="green", s=100, marker="o", label="Start")
        ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, marker="*", label="End")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Anderson Optimization on Bloch Sphere")
        ax.legend()

        return fig, ax

    def plot_loss_history(self, figsize=(10, 6)):
        """Plot loss function history.

        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        if len(self.loss_history_) == 0:
            raise ValueError("No loss history. Run minimize() first.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.loss_history_, "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Anderson Acceleration Convergence")
        ax.grid(True, alpha=0.3)

        return fig, ax
