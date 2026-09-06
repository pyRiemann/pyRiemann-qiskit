import numpy as np
import pytest
from docplex.mp.model import Model

from pyriemann_qiskit.optimization.pkit_optimizer import HAS_PKIT

# The parametrize decorators below reference PBitClassicalOptimizer /
# PBitQAOAOptimizer at collection time, before any skip marker can apply, so
# the test functions themselves must only be defined when p-kit is
# importable -- a pytest.mark.skipif alone is not enough to avoid a
# NameError when it is not installed.
if HAS_PKIT:
    from pyriemann_qiskit.optimization.pkit_optimizer import (
        PBitClassicalOptimizer,
        PBitQAOAOptimizer,
    )

    @pytest.mark.parametrize("optimizer", [PBitClassicalOptimizer, PBitQAOAOptimizer])
    def test_optimizer_creation(optimizer):
        assert optimizer()

    @pytest.mark.parametrize(
        "optimizer",
        [PBitClassicalOptimizer(upper_bound=1), PBitQAOAOptimizer(upper_bound=1)],
    )
    def test_solve_toy_qubo(optimizer):
        """Minimize -2*x0 - 2*x1 + x0*x1 over {0, 1}^2.

        Known optimum is x0 = x1 = 1 (value -3). This exercises the full
        docplex model -> QUBO -> p-bit circuit -> sample -> decode pipeline
        against a brute-force verifiable answer, rather than only checking
        that the classes can be instantiated.

        Uses each optimizer's default solver budget (Nt, n_shots, ...): a
        smaller budget was tried and converges less reliably across seeds,
        since CaSuDaSolver is a stochastic annealer, not an exact solver.
        """
        prob = Model()
        x0 = prob.binary_var(name="x0")
        x1 = prob.binary_var(name="x1")
        prob.minimize(-2 * x0 - 2 * x1 + x0 * x1)

        result = optimizer.solve(prob, reshape=False)

        np.testing.assert_array_equal(result, [1.0, 1.0])
