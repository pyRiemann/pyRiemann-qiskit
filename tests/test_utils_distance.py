from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from docplex.mp.model import Model
from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_logeuclid
from qiskit_addon_opt_mapper.translators import from_docplex_mp
from qiskit_algorithms.optimizers import SLSQP
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

import pyriemann_qiskit.optimization.docplex as docplex_module
from pyriemann_qiskit.classification import QuanticMDM
from pyriemann_qiskit.optimization.distance import (
    _get_bounds,
    qdistance_logeuclid_to_convex_hull,
    weights_logeuclid_to_convex_hull,
)
from pyriemann_qiskit.optimization.docplex import (
    ClassicalOptimizer,
    IntegerEncoding,
    NaiveQAOAOptimizer,
    QAOACVOptimizer,
    _reshape_solution,
    _to_qubo,
    pyQiskitOptimizer,
)
from pyriemann_qiskit.optimization.pkit_optimizer import (
    HAS_PKIT,
    PBitClassicalOptimizer,
    PBitTFIsingOptimizer,
)
from pyriemann_qiskit.optimization.simplex import (
    SingleExcitationHullOptimizer,
    _decode_counts,
    _single_excitation_circuit,
    _single_excitation_probabilities,
    _uniform_angles,
)
from pyriemann_qiskit.utils.dataset import get_mne_sample


class _ExactIntegerOptimizer(pyQiskitOptimizer):
    """Exhaustively solve tiny integer models as an exact QUBO reference."""

    def __init__(self, upper_bound):
        super().__init__(encoding=IntegerEncoding(upper_bound))

    def _solve_qp(self, qp, reshape=True):
        converter, qubo = _to_qubo(qp)
        n_vars = qubo.get_num_vars()
        candidates = product((0.0, 1.0), repeat=n_vars)
        best = min(
            candidates, key=lambda values: qubo.objective.evaluate(np.asarray(values))
        )
        solution = converter.interpret(np.asarray(best))
        return _reshape_solution(solution, reshape)


@pytest.mark.parametrize(
    "metric",
    [
        {"mean": "euclid", "distance": "qeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid_hull"},
    ],
)
def test_performance(metric):
    clf = make_pipeline(XdawnCovariances(), QuanticMDM(metric=metric, quantum=False))
    skf = StratifiedKFold(n_splits=3)
    covset, labels = get_mne_sample()
    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize(
    "optimizer",
    [
        ClassicalOptimizer(),
        # NaiveQAOAOptimizer(),
        # QAOACVOptimizer()
    ],
)
def test_qdistance_logeuclid_to_convex_hull(optimizer, get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)

    dist = qdistance_logeuclid_to_convex_hull(covmats, covmats[0], optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)

    covmean = mean_logeuclid(covmats)
    dist = qdistance_logeuclid_to_convex_hull(covmats, covmean, optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "optimizer",
    [
        ClassicalOptimizer(),
        # NaiveQAOAOptimizer(),
        # QAOACVOptimizer()
    ],
)
def test_weight_logeuclid_to_convex_hull(optimizer):
    X_0 = np.array([[0.9, 1.1], [0.9, 1.1]])
    X_1 = X_0 + 1
    X_train = np.stack((X_0, X_1))
    X_test = (X_0 + X_1) / 3
    weights = weights_logeuclid_to_convex_hull(X_train, X_test, optimizer=optimizer)
    distances = 1 - weights
    assert distances.argmin() == 0


@pytest.mark.parametrize(
    ("target", "expected_weights"),
    [
        (1.0, [1.0, 0.0]),
        (3.0, [0.5, 0.5]),
    ],
)
def test_integer_convex_hull_matches_exact_qubo(target, expected_weights):
    """Integer hull weights lie on the simplex at the selected resolution."""
    matrices = np.array([[[1.0]], [[9.0]]])
    optimizer = _ExactIntegerOptimizer(upper_bound=2)

    weights = weights_logeuclid_to_convex_hull(
        matrices, np.array([[target]]), optimizer=optimizer
    )

    np.testing.assert_allclose(weights, expected_weights, atol=1e-12)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(weights >= 0)


@pytest.mark.parametrize(
    ("target_value", "expected_weights"),
    [(1.0, [1.0, 0.0]), (3.0, [0.5, 0.5])],
)
def test_naive_qaoa_recovers_vertex_and_interior_points(
    monkeypatch, target_value, expected_weights
):
    """A deterministic QAOA result decodes vertex and interior hull points."""

    class ExactQAOA:
        def __init__(self, callback, **kwargs):
            self.callback = callback

        def compute_minimum_eigenvalue(self, operator):
            diagonal = np.diag(operator.to_matrix()).real
            basis_index = int(np.argmin(diagonal))
            bitstring = format(basis_index, f"0{operator.num_qubits}b")
            self.callback(1, None, float(diagonal[basis_index]), {})
            return SimpleNamespace(best_measurement={"bitstring": bitstring})

    monkeypatch.setattr(docplex_module, "QAOA", ExactQAOA)
    monkeypatch.setattr(
        docplex_module,
        "_get_quantum_instance",
        lambda _: SimpleNamespace(backend=object()),
    )
    monkeypatch.setattr(
        docplex_module, "generate_preset_pass_manager", lambda **kwargs: object()
    )
    matrices = np.array([[[1.0]], [[9.0]]])
    target = np.array([[target_value]])
    optimizer = NaiveQAOAOptimizer(upper_bound=2)

    weights = weights_logeuclid_to_convex_hull(matrices, target, optimizer)
    distance = distance_logeuclid(mean_logeuclid(matrices, weights), target)

    np.testing.assert_allclose(weights, expected_weights, atol=1e-12)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert distance == pytest.approx(0.0, abs=1e-10)
    assert len(optimizer.evaluated_values_) > 0


def test_naive_qaoa_multistart_selects_best_qubo_candidate(monkeypatch):
    """Seeded extra starts are evaluated and the best QUBO result is kept."""
    points = []

    class FakeQAOA:
        def __init__(self, initial_point, callback, **kwargs):
            points.append(np.asarray(initial_point))
            self.callback = callback

        def compute_minimum_eigenvalue(self, operator):
            bitstring = "1" if len(points) == 1 else "0"
            self.callback(len(points), None, float(len(points)), {})
            return SimpleNamespace(best_measurement={"bitstring": bitstring})

    monkeypatch.setattr(docplex_module, "QAOA", FakeQAOA)
    monkeypatch.setattr(
        docplex_module,
        "_get_quantum_instance",
        lambda _: SimpleNamespace(backend=object()),
    )
    monkeypatch.setattr(
        docplex_module, "generate_preset_pass_manager", lambda **kwargs: object()
    )
    model = Model()
    model.minimize(model.binary_var(name="decision"))
    optimizer = NaiveQAOAOptimizer(num_starts=2, seed=42)

    solution = optimizer.solve(model, reshape=False)

    np.testing.assert_array_equal(solution, [0])
    np.testing.assert_array_equal(points[0], [0.0, 0.0])
    np.testing.assert_allclose(
        points[1], np.random.default_rng(42).uniform(0, 2 * np.pi, size=2)
    )
    assert len(optimizer.evaluated_values_) == 2


def test_resolution_schedule_selects_best_exact_grid():
    """A deterministic resolution sweep selects an exactly representable mean."""
    matrices = np.array([[[1.0]], [[64.0]]])
    target = np.array([[4.0]])
    optimizer = _ExactIntegerOptimizer(upper_bound=2)

    weights = weights_logeuclid_to_convex_hull(
        matrices, target, optimizer, resolution_bounds=(2, 3)
    )

    np.testing.assert_allclose(weights, [2 / 3, 1 / 3], atol=1e-12)
    assert optimizer.upper_bound == 2
    assert optimizer.encoding.upper_bound == 2


@pytest.mark.parametrize(
    ("resolution_bounds", "expected"),
    [(None, (2, [2])), ((3, 2), (2, [3, 2]))],
)
def test_get_bounds_returns_configured_bound_and_schedule(resolution_bounds, expected):
    optimizer = _ExactIntegerOptimizer(upper_bound=2)
    assert _get_bounds(optimizer, resolution_bounds) == expected


def test_qaoacv_loss_includes_equality_constraint_penalty():
    model = Model()
    x = model.continuous_var(lb=0, ub=1)
    y = model.continuous_var(lb=0, ub=1)
    model.minimize(x * x + y * y)
    model.add_constraint(x + y == 1)
    qp = from_docplex_mp(model)
    objective = qp._objective
    constraints = list(qp.linear_constraints)

    _, _, penalty = QAOACVOptimizer.prepare_model(qp)
    loss = docplex_module._objective_with_penalty(
        objective, constraints, penalty, [0.2, 0.3]
    )

    expected = objective.evaluate([0.2, 0.3]) + penalty * (0.2 + 0.3 - 1) ** 2
    assert penalty > 0
    assert loss == pytest.approx(expected)


@pytest.mark.skipif(not HAS_PKIT, reason="p-kit is an optional dependency")
@pytest.mark.parametrize(
    "optimizer_class", [PBitClassicalOptimizer, PBitTFIsingOptimizer]
)
def test_pbit_convex_hull_weights_are_normalized(optimizer_class):
    """p-bit integer solutions are decoded onto the normalized hull simplex."""
    matrices = np.array([[[1.0]], [[9.0]]])
    if optimizer_class is PBitTFIsingOptimizer:
        optimizer = optimizer_class(
            upper_bound=2, Nt=100, n_shots=4, n_replicas=2, seed=42
        )
    else:
        optimizer = optimizer_class(upper_bound=2, Nt=100, n_shots=4, seed=42)

    weights = weights_logeuclid_to_convex_hull(matrices, np.array([[3.0]]), optimizer)

    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)


def test_qaoacv_returns_feasible_convex_hull_weights():
    """QAOA-CV evaluates the penalized model and returns simplex weights."""
    matrices = np.array([[[1.0]], [[9.0]]])
    target = np.array([[3.0]])
    optimizer = QAOACVOptimizer(n_reps=1, optimizer=SLSQP(maxiter=3))

    weights = weights_logeuclid_to_convex_hull(matrices, target, optimizer)

    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
    assert len(optimizer.y_) > 0
    assert optimizer.decoded_solution_.shape == (2,)
    assert np.all(np.isfinite(optimizer.decoded_solution_))

    repeated = QAOACVOptimizer(n_reps=1, optimizer=SLSQP(maxiter=3))
    repeated_weights = weights_logeuclid_to_convex_hull(matrices, target, repeated)
    np.testing.assert_allclose(weights, repeated_weights, atol=1e-12)


def test_resolution_bounds_require_integer_encoding():
    matrices = np.array([[[1.0]], [[9.0]]])
    with pytest.raises(ValueError, match="integer encoded optimizer"):
        weights_logeuclid_to_convex_hull(
            matrices,
            np.array([[3.0]]),
            QAOACVOptimizer(),
            resolution_bounds=(7, 5),
        )


def test_single_excitation_circuit_and_analytic_probabilities_agree():
    """Analytic one-excitation probabilities match the circuit statevector."""
    from qiskit.quantum_info import Statevector

    circuit, parameters = _single_excitation_circuit(5)
    cases = [
        (np.zeros(4), [1, 0, 0, 0, 0]),
        (_uniform_angles(5), [0.2] * 5),
    ]
    expected = np.array([0.37, 0.21, 0.19, 0.14, 0.09])
    angles = []
    remaining = 1.0
    for weight in expected[:-1]:
        angles.append(2 * np.arccos(np.sqrt(weight / remaining)))
        remaining -= weight
    cases.append((np.asarray(angles), expected))

    for angles, expected_weights in cases:
        state = Statevector(circuit.assign_parameters(dict(zip(parameters, angles))))
        decoded = np.array([state.probabilities()[1 << i] for i in range(5)])
        np.testing.assert_allclose(decoded, expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            _single_excitation_probabilities(angles), expected_weights, atol=1e-12
        )


def test_single_excitation_decoder_rejects_low_valid_shot_fraction():
    counts = {"001": 60, "010": 20, "100": 10, "000": 10}
    weights, fraction = _decode_counts(counts, 3, 0.5, 32)
    np.testing.assert_allclose(weights, [60 / 90, 20 / 90, 10 / 90])
    assert fraction == pytest.approx(0.9)

    weights, fraction = _decode_counts(counts, 3, 0.95, 32)
    assert weights is None
    assert fraction == pytest.approx(0.9)


def test_single_excitation_decoder_maps_qiskit_bit_order():
    counts = {"00100": 37, "00010": 21, "00001": 42, "11000": 5}
    weights, fraction = _decode_counts(counts, 5, 0.5, 32)
    np.testing.assert_allclose(weights, [0.42, 0.21, 0.37, 0, 0])
    assert fraction == pytest.approx(100 / 105)


def test_single_excitation_hull_exact_is_feasible_and_repeatable():
    """The generic optimizer interface returns its best feasible hull point."""
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    assert issubclass(SingleExcitationHullOptimizer, pyQiskitOptimizer)
    matrices = np.array([[[1.0]], [[9.0]]])
    target = np.array([[3.0]])
    solutions = []
    for _ in range(2):
        optimizer = SingleExcitationHullOptimizer(
            exact=True,
            optimizer=CobylaOptimizer(maxiter=150),
            min_valid_fraction=1.0,
        )
        weights = weights_logeuclid_to_convex_hull(matrices, target, optimizer)
        solutions.append(weights)
        np.testing.assert_array_equal(weights, optimizer.weights_)
        assert weights.sum() == pytest.approx(1, abs=1e-12)
        assert np.all(weights >= 0)
        assert optimizer.minimum_ == pytest.approx(
            distance_logeuclid(mean_logeuclid(matrices, weights), target) ** 2,
            abs=1e-10,
        )
        assert optimizer.valid_fraction_ == pytest.approx(1, abs=1e-12)
        assert optimizer.evaluations_ > 0
    np.testing.assert_array_equal(solutions[0], solutions[1])
    np.testing.assert_allclose(solutions[0], [0.5, 0.5], atol=1e-3)


def test_single_excitation_hull_honors_reshape_argument():
    from docplex.mp.model import Model

    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    model = Model()
    weights = model.continuous_var_list(4, lb=0)
    model.minimize(model.sum(weights[i] ** 2 for i in range(4)))
    model.add_constraint(model.sum(weights) == 1)
    optimizer = SingleExcitationHullOptimizer(
        exact=True, optimizer=CobylaOptimizer(maxiter=20)
    )

    assert optimizer.solve(model, reshape=True).shape == (2, 2)
    assert optimizer.solve(model, reshape=False).shape == (4,)


def test_single_excitation_hull_rejects_non_simplex_models():
    from docplex.mp.model import Model

    model = Model()
    weights = model.binary_var_list(2)
    model.minimize(model.sum(weights))
    model.add_constraint(model.sum(weights) == 1)

    optimizer = SingleExcitationHullOptimizer(exact=True)
    with pytest.raises(ValueError, match="continuous variables.*unit simplex"):
        optimizer.solve(model)

    model = Model()
    weights = model.continuous_var_list(2, lb=0)
    model.maximize(weights[0])
    model.add_constraint(model.sum(weights) == 1)
    with pytest.raises(ValueError, match="continuous variables.*unit simplex"):
        optimizer.solve(model)


def test_single_excitation_hull_rejects_unreliable_evaluations(monkeypatch):
    """No reliable evaluation raises instead of returning a fallback point."""
    import pyriemann_qiskit.optimization.simplex as simplex_module
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    monkeypatch.setattr(
        simplex_module, "_single_excitation_probabilities", lambda angles: None
    )
    matrices = np.array([[[1.0]], [[9.0]]])
    optimizer = SingleExcitationHullOptimizer(
        exact=True, optimizer=CobylaOptimizer(maxiter=3)
    )
    with pytest.raises(RuntimeError, match="enough one-excitation shots"):
        weights_logeuclid_to_convex_hull(matrices, np.array([[3.0]]), optimizer)


def test_single_excitation_hull_shot_sampler_is_feasible():
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    matrices = np.array([[[1.0]], [[9.0]]])
    optimizer = SingleExcitationHullOptimizer(
        optimizer=CobylaOptimizer(maxiter=5), shots=128, seed=42
    )
    weights = weights_logeuclid_to_convex_hull(matrices, np.array([[3.0]]), optimizer)

    np.testing.assert_array_equal(weights, optimizer.weights_)
    assert weights.sum() == pytest.approx(1, abs=1e-12)
    assert np.all(weights >= 0)
    assert optimizer.valid_fraction_ == pytest.approx(1, abs=1e-12)
