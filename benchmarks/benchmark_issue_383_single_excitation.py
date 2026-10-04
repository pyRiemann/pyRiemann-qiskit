"""Measure a hull-specific single-excitation variational circuit.

This opt-in experiment uses the original continuous Log-Euclidean hull
objective. Ideal shot sampling is drawn from exact statevector probabilities;
it does not model hardware noise.
"""

import argparse
import json
from time import perf_counter

import numpy as np
from pyriemann.datasets import make_matrices
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_logeuclid
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from scipy.optimize import differential_evolution, minimize

from pyriemann_qiskit.optimization.simplex import (
    _decode_counts,
    _single_excitation_circuit,
    _uniform_angles,
)


class _BudgetReached(Exception):
    pass


def _objective_data(matrices, target):
    logs = np.array([logm(matrix) for matrix in matrices])
    target_log = logm(target)
    gram = np.array([[np.trace(a @ b) for b in logs] for a in logs])
    linear = np.array([np.trace(target_log @ a) for a in logs])
    constant = float(np.trace(target_log @ target_log))
    return gram, linear, constant


def _decode(probabilities_or_counts, n_weights, min_valid_fraction, min_shots):
    counts = {
        format(index, f"0{n_weights}b"): value
        for index, value in enumerate(probabilities_or_counts)
        if value > 0
    }
    return _decode_counts(counts, n_weights, min_valid_fraction, min_shots)


def _run_case(
    matrices, target, case, mode, method, budget, starts, seed, shots, minimum
):
    algorithm_globals.random_seed = seed
    gram, linear, constant = _objective_data(matrices, target)
    n_weights = len(linear)
    circuit, parameters = _single_excitation_circuit(n_weights)
    rng = np.random.default_rng(seed)
    limits = [budget // starts + (index < budget % starts) for index in range(starts)]
    best = None
    evaluations = 0
    started = perf_counter()

    for index, limit in enumerate(limits):
        initial = (
            _uniform_angles(n_weights)
            if index == 0
            else rng.uniform(0, np.pi, n_weights - 1)
        )
        local_evaluations = 0

        def loss(angles):
            nonlocal best, evaluations, local_evaluations
            if local_evaluations >= limit:
                raise _BudgetReached()
            bound = circuit.assign_parameters(dict(zip(parameters, angles)))
            counts = Statevector(bound).probabilities()
            if mode == "shots":
                counts = rng.multinomial(shots, counts)
            weights, valid_fraction = _decode(
                counts, n_weights, minimum, 0 if mode == "exact" else 32
            )
            value = (
                1e6
                if weights is None
                else float(weights @ gram @ weights - 2 * linear @ weights + constant)
            )
            value = max(value, 0.0)
            local_evaluations += 1
            evaluations += 1
            if best is None or value < best["squared_distance"]:
                best = {
                    "squared_distance": value,
                    "angles": np.asarray(angles).tolist(),
                    "weights": None if weights is None else weights.tolist(),
                    "valid_fraction": valid_fraction,
                }
            return value

        try:
            if method == "DE":
                differential_evolution(
                    loss,
                    [(0, np.pi)] * (n_weights - 1),
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

    exact = Statevector(
        circuit.assign_parameters(dict(zip(parameters, best["angles"])))
    ).probabilities()
    exact_weights, exact_valid_fraction = _decode(exact, n_weights, minimum, 0)
    exact_squared = float(
        exact_weights @ gram @ exact_weights - 2 * linear @ exact_weights + constant
    )
    return {
        "case": case,
        "mode": mode,
        "optimizer": method,
        "seed": seed,
        "shots": shots if mode == "shots" else None,
        "budget": budget,
        "starts": starts,
        "evaluations": evaluations,
        "seconds": perf_counter() - started,
        "distance": float(np.sqrt(best["squared_distance"])),
        "exact_distance_at_best_angles": float(np.sqrt(max(exact_squared, 0))),
        "exact_weights_at_best_angles": exact_weights.tolist(),
        "exact_valid_fraction": exact_valid_fraction,
        "best": best,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("vertex", "interior", "skew", "both"), default="both"
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
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.budget < 1 or args.starts < 1 or args.starts > args.budget:
        parser.error("budget must be positive and at least as large as starts")
    if args.shots < 1:
        parser.error("shots must be positive")
    if not 0 < args.min_valid_fraction <= 1:
        parser.error("min-valid-fraction must be in (0, 1]")
    matrices = make_matrices(
        5, 3, "spd", np.random.RandomState(1234), return_params=False
    )
    cases = {
        "vertex": matrices[0],
        "interior": mean_logeuclid(matrices),
        "skew": mean_logeuclid(matrices, [0.05, 0.1, 0.2, 0.25, 0.4]),
    }
    for case, target in cases.items():
        if args.case == "both" or args.case == case:
            print(
                json.dumps(
                    _run_case(
                        matrices,
                        target,
                        case,
                        args.mode,
                        args.optimizer,
                        args.budget,
                        args.starts,
                        args.seed,
                        args.shots,
                        args.min_valid_fraction,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
