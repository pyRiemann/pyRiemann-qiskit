"""Optimizers backed by p-kit, a probabilistic-bit (p-bit) circuit simulator.

p-kit (https://github.com/IBM/p-kit) solves Ising/QUBO problems on classical
hardware using stochastic p-bit dynamics, as an alternative to running QAOA
on a quantum simulator or device. Note that it is an *alternative* solver,
not a reimplementation of QAOA: see :class:`PBitTFIsingOptimizer` for what it
does and does not share with QAOA. It is an *optional* dependency
of this package: install it separately (e.g.
``pip install git+https://github.com/IBM/p-kit.git``) to use the classes in
this module. If it is not installed, this module still imports cleanly, but
none of the classes below are defined.

Notes
-----
.. versionadded:: 0.7.0
"""

import math

import numpy as np
from qiskit_addon_opt_mapper.converters import EqualityToPenalty, IntegerToBinary

from .docplex import NaiveQAOAOptimizer, pyQiskitOptimizer

try:
    from joblib import Parallel, delayed
    from p_kit.library.poly import PolyOptimizer
    from p_kit.library.quantum import TransverseFieldIsing
    from p_kit.solver.annealing import constant
    from p_kit.solver.csd_solver import CaSuDaSolver

    HAS_PKIT = True
except ImportError:
    HAS_PKIT = False


if HAS_PKIT:

    def _sample_final_states(circuit, Nt, dt, i0, n_shots, seed, n_jobs):
        """Run `n_shots` independent p-bit annealing trajectories and return
        their final +/-1 states, shape (n_shots, n_pbits).

        Built directly on ``CaSuDaSolver`` rather than p-kit's own
        ``execute()`` helper: ``CaSuDaSolver.copy()`` reseeds its RNG from
        the *same* seed on every copy, so calling ``execute()`` with a fixed
        seed makes every one of its `n_shots` replay an identical
        trajectory -- no actual sampling diversity, which matters since
        CaSuDaSolver is a stochastic annealer that needs several independent
        reads to reliably find the minimum. This reseeds each shot
        explicitly instead, and reads only the final state of each run
        (``return_final=True``), which also skips storing the full
        trajectory.
        """

        def _run_one(shot_seed):
            solver = CaSuDaSolver(Nt=Nt, dt=dt, i0=i0, seed=shot_seed)
            return solver.solve(circuit, annealing_func=constant, return_final=True)

        seeds = [None if seed is None else seed + i for i in range(n_shots)]
        states = Parallel(n_jobs=n_jobs)(delayed(_run_one)(s) for s in seeds)
        return np.array(states)

    def _qubo_to_matrix(qubo):
        """Combine a QUBO's linear and quadratic objective terms into the
        single upper-triangular matrix expected by
        :func:`p_kit.library.quantum.TransverseFieldIsing.from_qubo`.
        """
        Q = qubo.objective.quadratic.to_array()
        np.fill_diagonal(Q, Q.diagonal() + qubo.objective.linear.to_array())
        return Q

    def _normalize_scale(max_abs_value):
        """Return a divisor bringing coefficients of magnitude
        `max_abs_value` down to O(1).

        ``CaSuDaSolver`` computes ``exp(-m * (field + h))`` at every step: raw
        docplex objectives (e.g. traces of matrix products) can carry
        coefficients in the thousands, which overflows that exponential and
        collapses the annealer to a numerical artifact instead of a proper
        minimum. Minimizing ``c * f(x)`` for ``c > 0`` has the same arg min as
        minimizing ``f(x)``, so rescaling the circuit's own J/h is free.
        """
        return max_abs_value if max_abs_value > 1 else 1.0

    def _best_candidate(objective, candidates):
        """Return the candidate minimizing `objective`.

        CaSuDaSolver is a stochastic sampler, not an exact solver: a batch of
        reads typically contains several distinct spin configurations, and
        the lowest-energy one (rather than an arbitrary or majority one) is
        kept as the solution.
        """
        best_x, best_value = None, None
        for x in candidates:
            value = objective.evaluate(x)
            if best_value is None or value < best_value:
                best_x, best_value = x, value
        return best_x

    class PBitClassicalOptimizer(pyQiskitOptimizer):
        """Wrapper for p-kit's classical (Boltzmann) p-bit optimizer.

        Encodes the docplex model directly with p-kit's
        :class:`~p_kit.library.poly.PolyOptimizer`, which performs its own
        integer-to-binary expansion, then samples the ground state with
        :class:`~p_kit.library.csd_solver.CaSuDaSolver`. This is the
        classical baseline: no transverse field, hence no emulated quantum
        dynamics -- see :class:`PBitTFIsingOptimizer` for the transverse-field
        counterpart.

        Parameters
        ----------
        upper_bound : int, default=7
            The maximum integer value for matrix normalization, and the
            value used to infer the bit-width of p-kit's binary expansion.
        Nt : int, default=10000
            Number of annealing timesteps run by the p-bit solver.
        dt : float, default=0.1
            Solver integration timestep.
        i0 : float, default=1.0
            Coupling-strength scale used by the (constant) annealing
            schedule. Larger values push the dynamics towards a more
            deterministic, greedy descent.
        n_shots : int, default=100
            Number of independent p-bit annealing trajectories to sample;
            each is run with its own seed (`seed` + shot index) so they
            explore independently, and the lowest-energy final state across
            all of them is kept as the solution.
        n_jobs : int, default=1
            Number of parallel jobs used to run the `n_shots` trajectories
            (via ``joblib``).
        seed : int, default=42
            Base seed for the p-bit solver's random number generator.

        Notes
        -----
        .. versionadded:: 0.7.0

        See Also
        --------
        pyQiskitOptimizer
        PBitTFIsingOptimizer
        """

        def __init__(
            self,
            upper_bound=7,
            Nt=10000,
            dt=0.1,
            i0=1.0,
            n_shots=100,
            n_jobs=1,
            seed=42,
        ):
            super().__init__()
            self.upper_bound = upper_bound
            self.Nt = Nt
            self.dt = dt
            self.i0 = i0
            self.n_shots = n_shots
            self.n_jobs = n_jobs
            self.seed = seed

        def convert_spdmat(self, X):
            return NaiveQAOAOptimizer.convert_spdmat(self, X)

        def spdmat_var(self, prob, channels, name):
            return NaiveQAOAOptimizer.spdmat_var(self, prob, channels, name)

        def get_weights(self, prob, classes):
            return NaiveQAOAOptimizer.get_weights(self, prob, classes)

        def _solve_qp(self, qp, reshape=True):
            qubo = EqualityToPenalty().convert(qp)
            variables = [v.name for v in qubo.variables]

            linear = qubo.objective.linear.to_array()
            quadratic = qubo.objective.quadratic.to_array()
            coeffs = {}
            for i, name_i in enumerate(variables):
                if linear[i]:
                    coeffs[(name_i,)] = coeffs.get((name_i,), 0) + linear[i]
                for j in range(i, len(variables)):
                    value = quadratic[i, j]
                    if not value:
                        continue
                    key = (
                        (name_i, name_i)
                        if i == j
                        else tuple(sorted((name_i, variables[j])))
                    )
                    coeffs[key] = coeffs.get(key, 0) + value

            scale = _normalize_scale(max((abs(v) for v in coeffs.values()), default=1))
            coeffs = {mono: value / scale for mono, value in coeffs.items()}

            n_bits = max(1, math.ceil(math.log2(self.upper_bound + 1)))
            circuit = PolyOptimizer(coeffs, variables, n_bits=n_bits, minimize=True)

            samples = _sample_final_states(
                circuit, self.Nt, self.dt, self.i0, self.n_shots, self.seed, self.n_jobs
            )

            candidates = [
                np.array([circuit.decode(row)[name] for name in variables], dtype=float)
                for row in samples
            ]
            result = _best_candidate(qubo.objective, candidates)

            if reshape:
                n_channels = int(math.sqrt(result.shape[0]))
                return np.reshape(result, (n_channels, n_channels))
            return result

    class PBitTFIsingOptimizer(pyQiskitOptimizer):
        """Wrapper for p-kit's transverse-field-Ising p-bit optimizer.

        Maps the docplex model to a QUBO (via ``IntegerToBinary`` +
        ``EqualityToPenalty``, same convention as
        :class:`~pyriemann_qiskit.optimization.docplex.NaiveQAOAOptimizer`),
        then builds a :class:`~p_kit.library.quantum.TransverseFieldIsing`
        p-bit circuit and samples it with
        :class:`~p_kit.library.csd_solver.CaSuDaSolver`.

        This is a *drop-in replacement* for
        :class:`~pyriemann_qiskit.optimization.docplex.NaiveQAOAOptimizer`,
        not a classical reimplementation of QAOA: the two share the
        problem, not the search. See the Notes below before interpreting
        any comparison between them.

        Parameters
        ----------
        upper_bound : int, default=7
            The maximum integer value for matrix normalization.
        gamma : float, default=0.3
            Transverse field strength of the emulated Ising model. Held
            fixed: unlike a QAOA mixer angle, it is not variationally
            optimized. ``gamma=0`` recovers the classical Ising limit (see
            :class:`PBitClassicalOptimizer` instead, which is cheaper for
            that case).
        beta : float, default=5.0
            Inverse temperature of the emulated quantum system. Lower
            values (e.g. 3.0) occasionally fail to settle on the true
            optimum within the default annealing budget, since
            ``CaSuDaSolver`` is a stochastic sampler rather than an exact
            solver; 5.0 was found empirically more robust across seeds for
            this reason.
        n_replicas : int, default=10
            Number of Suzuki-Trotter replicas: imaginary-time slices of the
            path-integral representation of the transverse-field Ising
            model. More replicas make the emulation of the quantum model
            more faithful (the Trotter error shrinks) at proportional
            cost. These are *not* QAOA layers (see Notes), so this is not a
            counterpart of ``NaiveQAOAOptimizer``'s `n_reps`.
        Nt : int, default=10000
            Number of annealing timesteps run by the p-bit solver.
        dt : float, default=0.1
            Solver integration timestep.
        i0 : float, default=1.0
            Coupling-strength scale used by the (constant) annealing
            schedule.
        n_shots : int, default=100
            Number of independent p-bit annealing trajectories to sample;
            each is run with its own seed (`seed` + shot index) so they
            explore independently, and the lowest-energy final state across
            all of them is kept as the solution.
        n_jobs : int, default=1
            Number of parallel jobs used to run the `n_shots` trajectories
            (via ``joblib``).
        seed : int, default=42
            Base seed for the p-bit solver's random number generator.

        Notes
        -----
        What this optimizer shares with
        :class:`~pyriemann_qiskit.optimization.docplex.NaiveQAOAOptimizer`
        is the problem: the same docplex model, mapped to the same
        QUBO / Ising cost Hamiltonian. How each one searches that problem
        differs substantially:

        * ``NaiveQAOAOptimizer`` builds an actual QAOA circuit -- `n_reps`
          layers of alternating cost and mixer unitaries -- and runs a
          *variational* classical outer loop (e.g. SLSQP) tuning the
          per-layer angles to minimize the measured expectation value.
        * This class builds one *static* transverse-field Ising model at
          fixed `gamma` and `beta` -- no variational angles, no outer loop
          -- in its Suzuki-Trotter (path-integral) representation, where
          the `n_replicas` copies are imaginary-time slices coupled along
          the Trotter axis, and samples its low-energy states with a
          stochastic p-bit annealer. This is the quantum Monte Carlo /
          stoquastic-Hamiltonian emulation regime described by Camsari et
          al. [1]_, the same family as simulated quantum annealing. The
          Trotter replicas are not QAOA circuit layers.

        The transverse field `gamma` is loosely comparable to QAOA's mixer
        only in that it is what makes the dynamics non-classical:
        ``gamma=0`` collapses the model to a plain classical Ising system.
        The resemblance ends there. A benchmark of this class against
        ``NaiveQAOAOptimizer`` therefore compares two different solvers on
        one shared problem; it is not an isolation of "the same QAOA, on
        different hardware".

        .. versionadded:: 0.7.0

        References
        ----------
        .. [1] K. Y. Camsari, S. Chowdhury and S. Datta,
               "Scaled Quantum Circuits Emulated with Room-Temperature
               p-Bits". Physical Review Applied, 2019.
               https://doi.org/10.1103/PhysRevApplied.12.034061

        See Also
        --------
        pyQiskitOptimizer
        PBitClassicalOptimizer
        pyriemann_qiskit.optimization.docplex.NaiveQAOAOptimizer
        """

        def __init__(
            self,
            upper_bound=7,
            gamma=0.3,
            beta=5.0,
            n_replicas=10,
            Nt=10000,
            dt=0.1,
            i0=1.0,
            n_shots=100,
            n_jobs=1,
            seed=42,
        ):
            super().__init__()
            self.upper_bound = upper_bound
            self.gamma = gamma
            self.beta = beta
            self.n_replicas = n_replicas
            self.Nt = Nt
            self.dt = dt
            self.i0 = i0
            self.n_shots = n_shots
            self.n_jobs = n_jobs
            self.seed = seed

        def convert_spdmat(self, X):
            return NaiveQAOAOptimizer.convert_spdmat(self, X)

        def spdmat_var(self, prob, channels, name):
            return NaiveQAOAOptimizer.spdmat_var(self, prob, channels, name)

        def get_weights(self, prob, classes):
            return NaiveQAOAOptimizer.get_weights(self, prob, classes)

        def _solve_qp(self, qp, reshape=True):
            conv = IntegerToBinary()
            qubo = conv.convert(qp)
            qubo = EqualityToPenalty().convert(qubo)

            Q = _qubo_to_matrix(qubo)
            n_bin = Q.shape[0]

            scale = _normalize_scale(np.max(np.abs(Q)) if Q.size else 1)
            circuit = TransverseFieldIsing.from_qubo(
                Q / scale, gamma=self.gamma, beta=self.beta, n_replicas=self.n_replicas
            )

            samples = _sample_final_states(
                circuit, self.Nt, self.dt, self.i0, self.n_shots, self.seed, self.n_jobs
            )

            # p-bit (i, tau) maps to index tau * n_bin + i (replica-major
            # ordering); average over the replica axis and threshold to
            # recover one classical bit per QUBO variable.
            spins = samples.reshape(-1, self.n_replicas, n_bin)
            bits = (spins.mean(axis=1) > 0).astype(float)

            best_bits = _best_candidate(qubo.objective, bits)
            result = conv.interpret(best_bits)

            if reshape:
                n_channels = int(math.sqrt(result.shape[0]))
                return np.reshape(result, (n_channels, n_channels))
            return result
