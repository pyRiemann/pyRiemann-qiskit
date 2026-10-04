"""Compare hull-specific simplex decoding with two QAOA-CV phase operators.

This is an opt-in experiment, not a solver backend. For example::

    python benchmarks/benchmark_issue_383_simplex.py --budget 250

Results are JSON lines so parameters and raw marginals can be retained.
The evaluation budget counts actual loss calls across all starts.
"""

import argparse
import json
from time import perf_counter

import numpy as np
from docplex.mp.model import Model
from pyriemann.datasets import make_matrices
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_logeuclid
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from qiskit_addon_opt_mapper.translators import from_docplex_mp
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from scipy.optimize import differential_evolution, minimize

from pyriemann_qiskit.optimization.docplex import QAOACVOptimizer


class _BudgetReached(Exception):
    pass


def _objective_data(matrices, target):
    logs = np.array([logm(matrix) for matrix in matrices])
    target_log = logm(target)
    gram = np.array([[np.trace(a @ b) for b in logs] for a in logs])
    linear = np.array([np.trace(target_log @ a) for a in logs])
    constant = np.trace(target_log @ target_log)
    return gram, linear, constant


def _circuit(gram, linear, phase):
    n_weights = len(linear)
    optimizer = QAOACVOptimizer()
    model = Model()
    weights = optimizer.get_weights(model, range(n_weights))
    model.minimize(
        model.sum(
            weights[i] * weights[j] * gram[i, j]
            for i in range(n_weights)
            for j in range(n_weights)
        )
        - 2 * model.sum(weights[i] * linear[i] for i in range(n_weights))
    )
    if phase == "penalty":
        model.add_constraint(model.sum(weights) == 1)
    qp, _ = QAOACVOptimizer.prepare_model(from_docplex_mp(model))
    cost, _ = qp.to_ising()
    mixer = optimizer.create_mixer(cost.num_qubits, use_params=True)
    return QAOAAnsatz(
        cost_operator=cost, reps=optimizer.n_reps, mixer_operator=mixer
    ).decompose()


def _marginals(circuit, parameters, selectors, shots, rng):
    probabilities = Statevector(circuit.assign_parameters(parameters)).probabilities()
    if shots is not None:
        probabilities = rng.multinomial(shots, probabilities) / shots
    return probabilities @ selectors


def _score(marginals, gram, linear, constant):
    marginal_sum = float(np.sum(marginals))
    if marginal_sum <= 1e-8:
        return 1e6, None
    weights = marginals / marginal_sum
    squared_distance = float(weights @ gram @ weights - 2 * linear @ weights + constant)
    return max(0.0, squared_distance), weights


def _run_case(matrices, target, case, phase, mode, method, budget, starts, seed, shots):
    algorithm_globals.random_seed = seed
    gram, linear, constant = _objective_data(matrices, target)
    circuit = _circuit(gram, linear, phase)
    n_weights = len(linear)
    selectors = (
        np.arange(2**n_weights)[:, None]
        & (1 << np.arange(n_weights - 1, -1, -1))[None, :]
    ) != 0
    rng = np.random.default_rng(seed)
    limits = [budget // starts + (index < budget % starts) for index in range(starts)]
    best = None
    evaluations = 0
    started = perf_counter()

    for index, limit in enumerate(limits):
        initial = (
            np.ones(circuit.num_parameters)
            if index == 0
            else rng.uniform(0, 2 * np.pi, circuit.num_parameters)
        )
        local_evaluations = 0

        def loss(parameters):
            nonlocal best, evaluations, local_evaluations
            if local_evaluations >= limit:
                raise _BudgetReached()
            marginals = _marginals(
                circuit,
                parameters,
                selectors,
                None if mode == "exact" else shots,
                rng,
            )
            value, weights = _score(marginals, gram, linear, constant)
            local_evaluations += 1
            evaluations += 1
            if best is None or value < best["loss"]:
                best = {
                    "loss": value,
                    "parameters": np.asarray(parameters).tolist(),
                    "raw_marginals": marginals.tolist(),
                    "weights": None if weights is None else weights.tolist(),
                }
            return value

        try:
            if method == "DE":
                differential_evolution(
                    loss,
                    [(0, 2 * np.pi)] * circuit.num_parameters,
                    seed=seed + index,
                    maxiter=1000,
                    popsize=5,
                    tol=0,
                    atol=0,
                    polish=False,
                )
            elif method == "spsa":
                SPSA(maxiter=max(1, limit // 2)).minimize(loss, initial)
            else:
                minimize(loss, initial, method=method, options={"maxiter": limit})
        except _BudgetReached:
            pass

    exact_raw = _marginals(
        circuit, np.asarray(best["parameters"]), selectors, None, rng
    )
    exact_value, exact_weights = _score(exact_raw, gram, linear, constant)
    training_raw = np.asarray(best["raw_marginals"])
    report = {
        "case": case,
        "phase": phase,
        "mode": mode,
        "optimizer": method,
        "seed": seed,
        "shots": None if mode == "exact" else shots,
        "budget": budget,
        "starts": starts,
        "evaluations": evaluations,
        "seconds": perf_counter() - started,
        "distance": float(np.sqrt(best["loss"])),
        "raw_sum": float(np.sum(best["raw_marginals"])),
        "raw_constraint_error": abs(float(np.sum(training_raw)) - 1),
        "raw_objective": float(
            training_raw @ gram @ training_raw - 2 * linear @ training_raw + constant
        ),
        "minimum_marginal": float(np.min(best["raw_marginals"])),
        "returned_sum_error": (
            None if best["weights"] is None else abs(sum(best["weights"]) - 1)
        ),
        "exact_distance_at_best_parameters": float(np.sqrt(exact_value)),
        "exact_raw_sum_at_best_parameters": float(np.sum(exact_raw)),
        "exact_weights_at_best_parameters": (
            None if exact_weights is None else exact_weights.tolist()
        ),
        "best": best,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("vertex", "interior", "skew", "both"), default="both"
    )
    parser.add_argument(
        "--phase", choices=("penalty", "no-penalty", "both"), default="both"
    )
    parser.add_argument("--mode", choices=("exact", "shots"), default="exact")
    parser.add_argument(
        "--optimizer",
        choices=("COBYLA", "SLSQP", "L-BFGS-B", "spsa", "DE"),
        default="COBYLA",
    )
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    if args.budget < 1 or args.starts < 1 or args.starts > args.budget:
        parser.error("budget must be positive and at least as large as starts")
    if args.shots < 1:
        parser.error("shots must be positive")
    matrices = make_matrices(
        5, 3, "spd", np.random.RandomState(1234), return_params=False
    )
    cases = {
        "vertex": matrices[0],
        "interior": mean_logeuclid(matrices),
        "skew": mean_logeuclid(matrices, [0.05, 0.1, 0.2, 0.25, 0.4]),
    }
    for case, target in cases.items():
        if args.case != "both" and args.case != case:
            continue
        for phase in ("penalty", "no-penalty"):
            if args.phase != "both" and args.phase != phase:
                continue
            print(
                json.dumps(
                    _run_case(
                        matrices,
                        target,
                        case,
                        phase,
                        args.mode,
                        args.optimizer,
                        args.budget,
                        args.starts,
                        args.seed,
                        args.shots,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
