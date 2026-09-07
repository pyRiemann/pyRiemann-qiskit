import contextlib

import numpy as np
import pytest
import qiskit_ibm_runtime
from docplex.mp.model import Model

import pyriemann_qiskit.classification.wrappers.quantic_classifier_base as base
from pyriemann_qiskit.classification import QuanticMDM, QuanticNCH
from pyriemann_qiskit.optimization.pkit_optimizer import HAS_PKIT


@pytest.mark.parametrize("classifier", [QuanticNCH, QuanticMDM])
def test_pkit_backend_does_not_init_qiskit(classifier, monkeypatch):
    """`backend="pkit"` must not set up any Qiskit backend.

    `quantum=True` is what selects PBitTFIsingOptimizer, but it also used to
    unconditionally build a Qiskit simulator/sampler -- and, with a
    `q_account_token`, to delete and re-save the user's IBM account and
    contact a provider -- even though nothing in the p-kit path uses any of
    it. Runs whether or not p-kit is installed: what is under test happens
    before any p-kit object is constructed.
    """
    touched = []

    for name in ("get_simulator", "get_provider", "get_device"):
        monkeypatch.setattr(
            base, name, (lambda n: lambda *a, **k: touched.append(n))(name)
        )
    monkeypatch.setattr(
        base, "BackendSamplerV2", lambda *a, **k: touched.append("BackendSamplerV2")
    )

    class _NoRuntimeService:
        @staticmethod
        def delete_account(*args, **kwargs):
            touched.append("delete_account")

        @staticmethod
        def save_account(*args, **kwargs):
            touched.append("save_account")

    monkeypatch.setattr(
        qiskit_ibm_runtime, "QiskitRuntimeService", _NoRuntimeService, raising=False
    )

    rs = np.random.RandomState(42)
    X = rs.randn(8, 3, 3)
    X = X @ X.transpose(0, 2, 1) + 3 * np.eye(3)
    y = np.array([0, 1] * 4)

    clf = classifier(
        quantum=True, backend="pkit", q_account_token="dummy_token", verbose=False
    )
    with contextlib.suppress(ImportError):
        # raised further down if p-kit is not installed, which is fine here:
        # by then the Qiskit-backend setup under test has already run.
        clf.fit(X, y)

    assert touched == []


# The parametrize decorators below reference PBitClassicalOptimizer /
# PBitTFIsingOptimizer at collection time, before any skip marker can apply, so
# what follows must only be defined when p-kit is importable -- a
# pytest.mark.skipif alone is not enough to avoid a NameError when it is not
# installed.
if HAS_PKIT:
    from pyriemann_qiskit.optimization.pkit_optimizer import (
        PBitClassicalOptimizer,
        PBitTFIsingOptimizer,
    )

    def _toy_model():
        """Minimize -2*x0 - 2*x1 + x0*x1 over {0, 1}^2; optimum x0 = x1 = 1."""
        prob = Model()
        x0 = prob.binary_var(name="x0")
        x1 = prob.binary_var(name="x1")
        prob.minimize(-2 * x0 - 2 * x1 + x0 * x1)
        return prob

    class TestPBitOptimizers:
        """Contract both p-bit optimizers must satisfy.

        Grouped as a class only to declare the optimizer parametrization once,
        via `pytestmark`, instead of repeating it on every test below. The
        annealing budget is kept well under each class's own defaults
        (Nt=10000, n_shots=100, ~7s per solve): Nt=500 with 10 shots hits the
        toy optimum on every seed tried and costs ~0.04s.
        """

        pytestmark = pytest.mark.parametrize(
            "optimizer", [PBitClassicalOptimizer, PBitTFIsingOptimizer]
        )

        def test_optimizer_creation(self, optimizer):
            assert optimizer()

        def test_solve_toy_qubo(self, optimizer):
            """Exercises the full docplex model -> QUBO -> p-bit circuit ->
            sample -> decode pipeline against a brute-force verifiable answer.
            """
            result = optimizer(upper_bound=1, Nt=500, n_shots=10, seed=0).solve(
                _toy_model(), reshape=False
            )
            np.testing.assert_array_equal(result, [1.0, 1.0])

        def test_seed_is_reproducible(self, optimizer):
            """A fixed seed must reproduce a run exactly.

            Guards the explicit per-shot reseeding in `_sample_final_states`,
            which exists because ``CaSuDaSolver.copy()`` would otherwise replay
            one identical trajectory for every shot.
            """
            first = optimizer(upper_bound=1, Nt=500, n_shots=10, seed=11).solve(
                _toy_model(), reshape=False
            )
            second = optimizer(upper_bound=1, Nt=500, n_shots=10, seed=11).solve(
                _toy_model(), reshape=False
            )
            np.testing.assert_array_equal(first, second)

        @pytest.mark.parametrize("upper_bound", [1, 3, 7])
        def test_upper_bound_respects_declared_domain(self, optimizer, upper_bound):
            """`upper_bound` must not widen a variable's declared domain.

            Both classes map the model with ``IntegerToBinary``, which expands
            each variable using its own bounds, so a binary model stays binary
            whatever `upper_bound` says. `PBitClassicalOptimizer` used to
            derive one bit-width from `upper_bound` and apply it to every
            variable, returning infeasible points such as [0, 7].
            """
            result = optimizer(
                upper_bound=upper_bound, Nt=500, n_shots=10, seed=0
            ).solve(_toy_model(), reshape=False)
            assert set(np.unique(result)) <= {0.0, 1.0}
            np.testing.assert_array_equal(result, [1.0, 1.0])

    @pytest.mark.parametrize("gamma", [0.0, 0.3, 1.0])
    def test_gamma_values(gamma):
        """`gamma=0` is the classical Ising limit, larger values raise the
        transverse field; the optimum must be found across that range.
        """
        result = PBitTFIsingOptimizer(
            upper_bound=1, gamma=gamma, Nt=500, n_shots=10, seed=0
        ).solve(_toy_model(), reshape=False)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    @pytest.mark.parametrize("n_replicas", [1, 2, 10])
    def test_n_replicas_values(n_replicas):
        """`n_replicas` sets the Suzuki-Trotter depth, and the solution is read
        back by averaging over that axis -- so the replica-major reshape in
        `_solve_qp` must hold for every value, including 1.
        """
        result = PBitTFIsingOptimizer(
            upper_bound=1, n_replicas=n_replicas, Nt=500, n_shots=10, seed=0
        ).solve(_toy_model(), reshape=False)
        np.testing.assert_array_equal(result, [1.0, 1.0])
