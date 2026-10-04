"""Single-excitation variational optimization for Log-Euclidean hulls."""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.primitives import BackendSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import SPSA

from ..utils.quantum_provider import get_simulator
from .docplex import _reshape_solution, pyQiskitOptimizer


def _single_excitation_circuit(n_weights):
    parameters = list(ParameterVector("theta", n_weights - 1))
    circuit = QuantumCircuit(n_weights)
    circuit.x(0)
    for index, parameter in enumerate(parameters):
        circuit.append(XXPlusYYGate(parameter), [index, index + 1])
    return circuit, parameters


def _uniform_angles(n_weights):
    return np.array(
        [
            2 * np.arccos(np.sqrt(1 / (n_weights - index)))
            for index in range(n_weights - 1)
        ]
    )


def _single_excitation_probabilities(angles):
    """Compute the one-excitation probabilities directly from circuit angles."""
    angles = np.asarray(angles, dtype=float)
    probabilities = np.empty(angles.size + 1)
    remaining = 1.0
    for index, angle in enumerate(angles):
        probabilities[index] = remaining * np.cos(angle / 2) ** 2
        remaining *= np.sin(angle / 2) ** 2
    probabilities[-1] = remaining
    return probabilities


def _decode_counts(counts, n_weights, min_valid_fraction, min_valid_shots):
    weights = np.zeros(n_weights)
    total = sum(counts.values())
    if total <= 0:
        return None, 0.0
    for bitstring, count in counts.items():
        bits = int(bitstring.replace(" ", ""), 2)
        if bits.bit_count() == 1:
            weights[bits.bit_length() - 1] += count
    valid = float(np.sum(weights))
    fraction = valid / total
    if valid < min_valid_shots or fraction + 1e-12 < min_valid_fraction:
        return None, fraction
    return weights / valid, fraction


class SingleExcitationHullOptimizer(pyQiskitOptimizer):
    """Optimize Log-Euclidean hull weights in a one-excitation state.

    The circuit contains one excited qubit and ``n_matrices - 1`` exchange
    rotations. Its ideal measurement probabilities are simplex weights. In
    the chain, ``w[0] = cos(theta[0] / 2)**2`` and each later weight is the
    remaining probability multiplied by ``cos(theta[i] / 2)**2``. Given any
    simplex vector, setting ``theta[i] = 2*arccos(sqrt(w[i] / remaining))``
    recursively represents it, with an arbitrary later angle after the
    remaining probability reaches zero. This establishes coverage of the
    full simplex with ``n_matrices - 1`` angles.

    Exact probabilities are computed directly from the exchange angles, with
    work linear in the number of weights. In shot mode, only measured
    bitstrings with exactly one ``1`` contribute to the weights. The returned
    weights are normalized conditionally on those valid shots. Too few valid
    shots mark an evaluation as unreliable; if no reliable evaluation occurs,
    :meth:`solve` raises ``RuntimeError``.

    Parameters
    ----------
    optimizer : qiskit_algorithms.optimizers.Optimizer or None, default=None
        Classical optimizer for circuit angles. ``None`` uses SPSA with 125
        iterations.
    shots : int, default=1024
        Measurement shots per loss evaluation in shot mode.
    exact : bool, default=False
        Compute exact probabilities analytically instead of sampling. This is
        useful for reproducible simulations.
    quantum_instance : BackendSamplerV2 or None, default=None
        Sampler for shot mode. ``None`` uses the configured local simulator.
    seed : int, default=42
        Seed for the default local sampler.
    min_valid_fraction : float, default=0.5
        Minimum fraction of one-excitation shots for a reliable evaluation.
    min_valid_shots : int, default=32
        Minimum count of one-excitation shots for a reliable evaluation.

    Attributes
    ----------
    weights_ : ndarray
        Returned feasible weights from the best loss evaluation. In shot mode,
        these are conditional weights normalized over valid one-excitation
        shots.
    valid_fraction_ : float
        Fraction of valid shots in that evaluation.
    optim_params_ : ndarray
        Circuit parameters of that evaluation.
    minimum_ : float
        Squared Log-Euclidean distance of the returned weights.
    evaluations_ : int
        Number of loss evaluations.
    run_time_ : float
        Optimization time in seconds.

    See Also
    --------
    pyriemann_qiskit.optimization.docplex.pyQiskitOptimizer
    """

    def __init__(
        self,
        optimizer=None,
        shots=1024,
        exact=False,
        quantum_instance=None,
        seed=42,
        min_valid_fraction=0.5,
        min_valid_shots=32,
    ):
        super().__init__()
        if shots < 1 or min_valid_shots < 1:
            raise ValueError("shots and min_valid_shots must be positive")
        if not exact and min_valid_shots > shots:
            raise ValueError("min_valid_shots cannot exceed shots")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        self.optimizer = SPSA(maxiter=125) if optimizer is None else optimizer
        self.shots = shots
        self.exact = exact
        self.quantum_instance = quantum_instance
        self.seed = seed
        self.min_valid_fraction = min_valid_fraction
        self.min_valid_shots = min_valid_shots

    def _solve_qp(self, qp, reshape=True):
        """Solve a continuous simplex model using the exchange circuit."""
        n_weights = qp.get_num_vars()
        variables = qp.variables
        constraints = qp.linear_constraints
        is_continuous = all(
            variable.vartype.name == "CONTINUOUS" for variable in variables
        )
        bounds_cover_simplex = all(
            (variable.lowerbound is None or variable.lowerbound <= 0)
            and (variable.upperbound is None or variable.upperbound >= 1)
            for variable in variables
        )
        is_unit_simplex = (
            len(constraints) == 1
            and constraints[0].sense == constraints[0].Sense.EQ
            and np.isclose(constraints[0].rhs, 1)
            and np.allclose(constraints[0].linear.to_array(), np.ones(n_weights))
        )
        if (
            not is_continuous
            or not bounds_cover_simplex
            or not is_unit_simplex
            or qp.objective.sense != qp.objective.Sense.MINIMIZE
            or qp.objective.higher_order
        ):
            raise ValueError(
                "SingleExcitationHullOptimizer supports minimization models "
                "with continuous variables on the unit simplex and a "
                "linear or quadratic objective"
            )
        if n_weights < 2:
            raise ValueError("single-excitation hull needs at least two matrices")
        circuit, parameters = _single_excitation_circuit(n_weights)
        if self.exact:
            measured_circuit = None
            sampler = None
        else:
            sampler = self.quantum_instance
            if sampler is None:
                sampler = BackendSamplerV2(
                    backend=get_simulator(),
                    options={
                        "default_shots": self.shots,
                        "seed_simulator": self.seed,
                    },
                )
            pass_manager = generate_preset_pass_manager(
                optimization_level=1, backend=sampler.backend
            )
            measured_circuit = circuit.copy()
            measured_circuit.measure_all()
            measured_circuit = pass_manager.run(measured_circuit)

        self.evaluations_ = 0
        best = None

        def loss(angles):
            nonlocal best
            binding = dict(zip(parameters, angles))
            if self.exact:
                weights = _single_excitation_probabilities(angles)
                valid_fraction = 1.0
            else:
                bound = measured_circuit.assign_parameters(binding)
                job = sampler.run([bound], shots=self.shots)
                counts = job.result()[0].data.meas.get_counts()
                weights, valid_fraction = _decode_counts(
                    counts,
                    n_weights,
                    self.min_valid_fraction,
                    self.min_valid_shots,
                )
            self.evaluations_ += 1
            if weights is None:
                return 1e6
            value = float(qp.objective.evaluate(weights))
            for constraint in qp.linear_constraints:
                violation = constraint.evaluate(weights) - constraint.rhs
                if constraint.sense == constraint.Sense.EQ:
                    value += 1e5 * violation**2
                elif constraint.sense == constraint.Sense.LE:
                    value += 1e5 * max(violation, 0) ** 2
                else:
                    value += 1e5 * min(violation, 0) ** 2
            if best is None or value < best[0]:
                best = (
                    value,
                    weights.copy(),
                    valid_fraction,
                    np.asarray(angles).copy(),
                )
            return value

        start = time.perf_counter()
        self.optimizer.minimize(loss, _uniform_angles(n_weights), bounds=None)
        self.run_time_ = time.perf_counter() - start
        if best is None:
            raise RuntimeError("no evaluation had enough one-excitation shots")
        self.minimum_, self.weights_, self.valid_fraction_, self.optim_params_ = best
        return _reshape_solution(self.weights_.copy(), reshape)
