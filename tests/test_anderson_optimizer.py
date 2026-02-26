import numpy as np
import pytest

from pyriemann_qiskit.utils.anderson_optimizer import AndersonAccelerationOptimizer


def quadratic(x):
    return np.sum(x**2)


def test_convergence():
    """Optimizer converges on a simple quadratic."""
    opt = AndersonAccelerationOptimizer(maxiter=30, m=5)
    result = opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    assert result.fun < 1e-6


def test_trajectory_length():
    """trajectory_ has one entry per iteration plus the initial point."""
    opt = AndersonAccelerationOptimizer(maxiter=30, m=5, tol=0.0)
    opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    # initial point + 30 iterations = 31
    assert len(opt.trajectory_) == 31


def test_loss_history_length():
    """loss_history_ matches trajectory_ length."""
    opt = AndersonAccelerationOptimizer(maxiter=30, m=5, tol=0.0)
    opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    assert len(opt.loss_history_) == len(opt.trajectory_)


def test_trajectory_first_entry_is_x0():
    """First trajectory entry equals x0."""
    x0 = np.array([1.0, 2.0, 3.0])
    opt = AndersonAccelerationOptimizer(maxiter=10, m=5)
    opt.minimize(quadratic, x0=x0)
    np.testing.assert_array_equal(opt.trajectory_[0], x0)


def test_result_fields():
    """Result object has expected fields."""
    opt = AndersonAccelerationOptimizer(maxiter=10, m=5)
    result = opt.minimize(quadratic, x0=np.array([1.0, 0.5]))
    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert hasattr(result, "nfev")
    assert hasattr(result, "nit")
    assert result.nfev > 0
    assert result.nit > 0


def test_nfev_count():
    """nfev equals 1 (initial) + maxiter * (2n + 1) at most."""
    n = 3
    maxiter = 5
    opt = AndersonAccelerationOptimizer(maxiter=maxiter, m=5, tol=0.0)
    result = opt.minimize(quadratic, x0=np.ones(n))
    # Each iteration: 2n grad evals + 1 update eval
    assert result.nfev == 1 + maxiter * (2 * n + 1)


def test_with_periodic_bounds():
    """Optimizer runs without error when periodic bounds are provided."""
    bounds = [(0, 2 * np.pi)] * 2
    opt = AndersonAccelerationOptimizer(maxiter=20, m=5)
    result = opt.minimize(quadratic, x0=np.array([0.5, 1.0]), bounds=bounds)
    assert result.fun >= 0
    # Result stays within bounds
    assert np.all(result.x >= 0)
    assert np.all(result.x <= 2 * np.pi)


def test_settings():
    """settings property returns correct values."""
    opt = AndersonAccelerationOptimizer(maxiter=50, m=7, alpha=0.5, tol=1e-4)
    s = opt.settings
    assert s["maxiter"] == 50
    assert s["m"] == 7
    assert s["alpha"] == 0.5
    assert s["tol"] == 1e-4
