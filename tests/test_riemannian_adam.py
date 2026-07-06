import numpy as np

from pyriemann_qiskit.optimization.riemannian_adam import (
    RiemannianAdamOptimizer,
)


def quadratic(x):
    return np.sum(x**2)


def test_convergence():
    """Optimizer converges on a simple quadratic."""
    opt = RiemannianAdamOptimizer(maxiter=200, lr=0.1)
    result = opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    assert result.fun < 1e-3


def test_trajectory_length():
    """trajectory_ has one entry per iteration plus the initial point."""
    opt = RiemannianAdamOptimizer(maxiter=30, tol=0.0)
    opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    # initial point + 30 iterations = 31
    assert len(opt.trajectory_) == 31


def test_loss_history_length():
    """loss_history_ matches trajectory_ length."""
    opt = RiemannianAdamOptimizer(maxiter=30, tol=0.0)
    opt.minimize(quadratic, x0=np.array([1.0, 2.0, 3.0]))
    assert len(opt.loss_history_) == len(opt.trajectory_)


def test_trajectory_first_entry_is_x0():
    """First trajectory entry equals x0."""
    x0 = np.array([1.0, 2.0, 3.0])
    opt = RiemannianAdamOptimizer(maxiter=10)
    opt.minimize(quadratic, x0=x0)
    np.testing.assert_array_equal(opt.trajectory_[0], x0)


def test_result_fields():
    """Result object has expected fields."""
    opt = RiemannianAdamOptimizer(maxiter=10)
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
    opt = RiemannianAdamOptimizer(maxiter=maxiter, tol=0.0)
    result = opt.minimize(quadratic, x0=np.ones(n))
    # Each iteration: 2n grad evals + 1 update eval
    assert result.nfev == 1 + maxiter * (2 * n + 1)


def test_with_periodic_bounds():
    """Optimizer wraps periodic (2*pi span) bounds instead of escaping them."""
    bounds = [(0, 2 * np.pi)] * 2
    opt = RiemannianAdamOptimizer(maxiter=50, lr=0.5)
    result = opt.minimize(quadratic, x0=np.array([0.5, 1.0]), bounds=bounds)
    assert result.fun >= 0
    assert np.all(result.x >= 0)
    assert np.all(result.x <= 2 * np.pi)


def test_with_clipped_bounds():
    """Non-periodic bounds (e.g. QIOCE's [0, pi]) are clipped, not wrapped."""
    bounds = [(0, np.pi)] * 2
    opt = RiemannianAdamOptimizer(maxiter=50, lr=0.5)
    result = opt.minimize(quadratic, x0=np.array([0.5, 1.0]), bounds=bounds)
    assert np.all(result.x >= 0)
    assert np.all(result.x <= np.pi)


def test_settings():
    """settings property returns correct values."""
    opt = RiemannianAdamOptimizer(maxiter=50, lr=0.05, beta1=0.8, tol=1e-4)
    s = opt.settings
    assert s["maxiter"] == 50
    assert s["lr"] == 0.05
    assert s["beta1"] == 0.8
    assert s["tol"] == 1e-4
