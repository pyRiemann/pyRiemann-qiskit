"""This module contains both classic and quantum optimizers, and some helper
functions. The quantum optimizer allows an optimization problem with
constraints (in the form of docplex model) to be run on a quantum computer.
It is for example suitable for:
- MDM optimization problem;
- computation of matrices mean.
"""
import math
import time

import numpy as np
from docplex.mp.vartype import BinaryVarType, ContinuousVarType, IntegerVarType
from pyriemann.utils.covariance import normalize
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import BackendSampler
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, SPSA
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters import IntegerToBinary, LinearEqualityToPenalty
from qiskit_optimization.problems import VarType
from qiskit_optimization.translators import from_docplex_mp
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import deprecated

from .hyper_params_factory import create_mixer_rotational_X_gates
from .math import is_pauli_identity
from .quantum_provider import get_simulator


@deprecated("set_global_optimizer is deprecated and will be removed in 0.6.0; ")
def set_global_optimizer():
    pass


@deprecated("get_global_optimizer is deprecated and will be removed in 0.6.0; ")
def get_global_optimizer():
    pass


def square_cont_mat_var(prob, channels, name="cont_spdmat"):
    """ Docplex square continuous matrix

    Creates a 2-dimensional dictionary of continuous decision variables,
    indexed by pairs of key objects.
    The dictionary represents a square matrix of size
    len(channels) x len(channels).
    A key can be any Python object, with the exception of None.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        A custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    square_mat : dict
        A square matrix of continuous decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    ContinuousVarType.one_letter_symbol = lambda _: "C"
    return prob.continuous_var_matrix(
        keys1=channels, keys2=channels, name=name, lb=-prob.infinity
    )


def square_int_mat_var(prob, channels, upper_bound=7, name="int_spdmat"):
    """ Docplex square integer matrix

    Creates a 2-dimensional dictionary of integer decision variables,
    indexed by pairs of key objects.
    The dictionary represents a square matrix of size
    len(channels) x len(channels).
    A key can be any Python object, with the exception of None.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    upper_bound : int (default: 7)
        The upper bound of the integer docplex variables.
    name : string
        A custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    square_mat : dict
        A square matrix of integer decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    IntegerVarType.one_letter_symbol = lambda _: "I"
    return prob.integer_var_matrix(
        keys1=channels, keys2=channels, name=name, lb=0, ub=upper_bound
    )


def square_bin_mat_var(prob, channels, name="bin_spdmat"):
    """ Docplex square binary matrix

    Creates a 2-dimensional dictionary of binary decision variables,
    indexed by pairs of key objects.
    The dictionary represents a square matrix of size
    len(channels) x len(channels).
    A key can be any Python object, with the exception of None.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        A custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    square_mat : dict
        A square matrix of binary decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    BinaryVarType.one_letter_symbol = lambda _: "B"
    return prob.binary_var_matrix(keys1=channels, keys2=channels, name=name)


class pyQiskitOptimizer:

    """Wrapper for Qiskit optimizer.

    This class is an abstract class which provides an interface
    for running a docplex model independently of the optimizer type
    (such as classical or quantum optimizer).

    Notes
    -----
    .. versionadded:: 0.0.2
    .. versionchanged:: 0.0.4
    """

    def __init__(self):
        pass

    def convert_spdmat(self, X):
        """Convert a SPD matrix

        Hook to apply some transformation on a SPD matrix.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_features)
            A SPD matrix.

        Returns
        -------
        X_new : ndarray, shape (n_features, n_features)
            A transformation of the SPD matrix.

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.4.0
            rename convert_covmat to convert_spdmat
        """
        return X

    def spdmat_var(self, prob, channels, name):
        """ Create docplex matrix variable

        Helper to create a docplex representation of a
        SPD matrix variable.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_.
        channels : list
            The list of channels. A channel can be any Python object,
            such as channels'name or number but None.
        name : string
            A custom name for the variable. The name is used internally by docplex
            and may appear if your print the model to a file for example.

        Returns
        -------
        docplex_spdmat : dict
            A docplex representation of a SPD matrix.

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.4.0
            rename covmat_var to spdmat_var

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

        """
        raise NotImplementedError()

    def _solve_qp(self, qp, reshape=True):
        raise NotImplementedError()

    def solve(self, prob, reshape=True):
        """Solve the docplex problem.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_

        Returns
        -------
        result : OptimizationResult
            The result of the optimization.

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.0.4

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

        """
        qp = from_docplex_mp(prob)
        return self._solve_qp(qp, reshape)

    def get_weights(self, prob, classes):
        """Weights variable

        Helper to create a docplex representation of a
        weight vector.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        classes : list
            The classes.

        Returns
        -------
        docplex_weights : dict
            A vector of decision variables representing
            weights.

        Notes
        -----
        .. versionadded:: 0.0.4

        """
        raise NotImplementedError()


class ClassicalOptimizer(pyQiskitOptimizer):

    """Wrapper for the classical Cobyla optimizer.

    Attributes
    ----------
    optimizer : OptimizationAlgorithm
        An instance of OptimizationAlgorithm [1]_

    Notes
    -----
    .. versionadded:: 0.0.2
    .. versionchanged:: 0.0.4
    .. versionchanged:: 0.2.0
        Add attribute `optimizer`.

    See Also
    --------
    pyQiskitOptimizer

    References
    ----------
    .. [1] \
        https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.OptimizationAlgorithm.html#optimizationalgorithm

    """

    def __init__(self, optimizer=CobylaOptimizer(rhobeg=2.1, rhoend=0.000001)):
        pyQiskitOptimizer.__init__(self)
        self.optimizer = optimizer

    def spdmat_var(self, prob, channels, name):
        """ Create docplex matrix variable

        Helper to create a docplex representation of a
        SPD matrix variable.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        channels : list
            The list of channels. A channel can be any Python object,
            such as channels'name or number but None.
        name : string
            A custom name for the variable. The name is used internally by docplex
            and may appear if your print the model to a file for example.

        Returns
        -------
        docplex_spdmat : dict
            A docplex representation of a SPD matrix with continuous variables.

        See Also
        -----
        square_cont_mat_var

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.4.0
            rename covmat_var to spdmat_var

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
        """
        return square_cont_mat_var(prob, channels, name)

    def _solve_qp(self, qp, reshape=True):
        result = self.optimizer.solve(qp).x
        if reshape:
            n_channels = int(math.sqrt(result.shape[0]))
            return np.reshape(result, (n_channels, n_channels))
        return result

    def get_weights(self, prob, classes):
        """Weights variabpe

        Helper to create a docplex representation of a
        weight vector.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        classes : list
            The classes.

        Returns
        -------
        docplex_weights : dict
            A vector of continuous decision variables representing weights.

        Notes
        -----
        .. versionadded:: 0.0.4

        """
        w = prob.continuous_var_matrix(
            keys1=[1], keys2=classes, name="weight", lb=0, ub=1
        )
        w = np.array([w[key] for key in w])
        return w


def _get_quantum_instance(self):
    if self.quantum_instance is None:
        backend = get_simulator()
        seed = 42
        shots = 1024
        quantum_instance = BackendSampler(
            backend, options={"shots": shots, "seed_simulator": seed}
        )
        quantum_instance.transpile_options["seed_transpiler"] = seed
    else:
        quantum_instance = self.quantum_instance
    return quantum_instance


class NaiveQAOAOptimizer(pyQiskitOptimizer):

    """Wrapper for the quantum optimizer QAOA.

    Parameters
    ----------
    upper_bound : int, default=7
        The maximum integer value for matrix normalization.
    quantum_instance : QuantumInstance, default=None
        A quantum backend instance.
        If None, AerSimulator will be used.
    optimizer : SciPyOptimizer, default=SLSQP()
        An instance of a scipy optimizer to find the optimal weights for the
        parametric circuit (ansatz).
    initial_points : Tuple[int, int], default=[0.0, 0.0].
        Starting parameters (beta and gamma) for the QAOA.

    Notes
    -----
    .. versionadded:: 0.0.2
    .. versionchanged:: 0.0.4
        add get_weights method.
    .. versionchanged:: 0.3.0
        add `evaluated_values_` attribute.
        add optimizer parameter.

    Attributes
    ----------
    evaluated_values_ : list[int]
        Training curve values.

    See Also
    --------
    pyQiskitOptimizer
    """

    def __init__(
        self,
        upper_bound=7,
        quantum_instance=None,
        optimizer=SLSQP(),
        initial_points=[0.0, 0.0],
    ):
        pyQiskitOptimizer.__init__(self)
        self.upper_bound = upper_bound
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer
        self.initial_points = initial_points

    def convert_spdmat(self, X):
        """Convert a SPD matrix

        Transform all values in the SPD matrix to integers.

        Example:
        0.123 -> 1230

        Parameters
        ----------
        X : ndarray, shape (n_features, n_features)
            The SPD matrix.

        Returns
        -------
        transformed_X : ndarray, shape (n_features, n_features)
            A transformation of the SPD matrix.

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.4.0
            rename convert_covmat to convert_spdmat

        """
        corr = normalize(X, "corr")
        return np.round(corr * self.upper_bound, 0)

    def spdmat_var(self, prob, channels, name):
        """ Create docplex matrix variable

        Helper to create a docplex representation of a
        SPD matrix variable.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        channels : list
            The list of channels. A channel can be any Python object,
            such as channels'name or number but None.
        name : string
            A custom name for the variable. The name is used internally by docplex
            and may appear if your print the model to a file for example.

        Returns
        -------
        docplex_spdmat : dict
            A docplex representation of a SPD matrix with integer variables.

        See Also
        -----
        square_int_mat_var

        Notes
        -----
        .. versionadded:: 0.0.2
        .. versionchanged:: 0.4.0
            rename covmat_var to spdmat_var

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

        """
        return square_int_mat_var(prob, channels, self.upper_bound, name)

    def _solve_qp(self, qp, reshape=True):
        conv = IntegerToBinary()
        qubo = conv.convert(qp)
        quantum_instance = _get_quantum_instance(self)

        self.evaluated_values_ = []

        def _callback(_eval_count, _weights, value, _meta):
            self.evaluated_values_.append(value)

        qaoa_mes = QAOA(
            sampler=quantum_instance,
            optimizer=self.optimizer,
            initial_point=self.initial_points,
            callback=_callback,
        )
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = conv.interpret(qaoa.solve(qubo))
        if reshape:
            n_channels = int(math.sqrt(result.shape[0]))
            return np.reshape(result, (n_channels, n_channels))
        return result

    def get_weights(self, prob, classes):
        """Get weights variable

        Helper to create a docplex representation of a weight vector.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        classes : list
            The classes.

        Returns
        -------
        docplex_weights : dict
            A vector of integer decision variables representing weights.

        Notes
        -----
        .. versionadded:: 0.0.4

        """
        w = prob.integer_var_matrix(
            keys1=[1], keys2=classes, name="weight", lb=0, ub=self.upper_bound
        )
        w = np.array([w[key] for key in w])
        return w


class QAOACVAngleOptimizer(pyQiskitOptimizer):

    """QAOA with continuous variables encoded in state vector angles.

    This optimizer encodes continuous variables directly in the angles/phases
    of the quantum state vector, rather than converting them to binary variables
    and using probability distributions.

    Parameters
    ----------
    create_mixer : Callable[int, QuantumCircuit], \
            default=create_mixer_rotational_X_gates(0)
        A delegate that takes into input an angle and returns a QuantumCircuit.
    n_reps : int, default=3
        The number of repetitions for the QAOA ansatz.
        It defines how many time Mixer and Cost operators will be repeated
        in the circuit.
    quantum_instance : QuantumInstance, default=None
        A quantum backend instance.
        If None, AerSimulator will be used.
    optimizer : SciPyOptimizer, default=L_BFGS_B(maxiter=100, maxfun=200)
        An instance of a scipy optimizer to find the optimal weights for the
        parametric circuit (ansatz).

    Notes
    -----
    .. versionadded:: 0.5.0

    Attributes
    ----------
    x_: list[int]
        Indices of the loss function.
    y_: list[int]
        Cost computed by the loss function.
    run_time_: float
        Time taken by the optimizer
    optim_params_: list[float]
        The optimal rotation for the gate in the circuit.
    minimum_: float
        The value of the objective function with the optimal parameters.
    state_vector_: StateVector
        State vector of the optimized quantum circuit
        (optimal parameters assigned to the parametric gates).
    variable_bounds_: list[tuple]
        The bounds for each continuous variable.

    See Also
    --------
    pyQiskitOptimizer
    QAOACVOptimizer
    create_mixer_rotational_X_gates
    """

    def __init__(
        self,
        create_mixer=create_mixer_rotational_X_gates(0),
        n_reps=3,
        quantum_instance=None,
        optimizer=L_BFGS_B(maxiter=100, maxfun=200),
    ):
        self.n_reps = n_reps
        self.create_mixer = create_mixer
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer
        print("Warning! QAOACVAngleOptimizer only support simulation")

    @staticmethod
    def prepare_model(qp):
        """Prepare model by storing variable bounds.

        Unlike QAOACVOptimizer, we don't convert to binary variables.
        We keep continuous variables and store their bounds for angle mapping.
        """
        variable_bounds = []
        for v in qp.variables:
            variable_bounds.append((v.lowerbound, v.upperbound))

        return variable_bounds

    def spdmat_var(self, prob, channels, name):
        """ Create docplex matrix variable

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_.
        channels : list
            The list of channels. A channel can be any Python object,
            such as channels'name or number but None.
        name : string
            A custom name for the variable. The name is used internally by docplex
            and may appear if your print the model to a file for example.

        Returns
        -------
        docplex_spdmat : dict
            A docplex representation of a SPD matrix with continuous variables.

        See Also
        -----
        square_cont_mat_var

        Notes
        -----
        .. versionadded:: 0.5.0

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

        """
        return ClassicalOptimizer.spdmat_var(self, prob, channels, name)

    def get_weights(self, prob, classes):
        """Get weights variable

        Helper to create a docplex representation of a weight vector.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        classes : list
            The classes.

        Returns
        -------
        docplex_weights : dict
            A vector of continuous decision variables representing weights.

        Notes
        -----
        .. versionadded:: 0.5.0

        """
        return ClassicalOptimizer.get_weights(self, prob, classes)

    def _solve_qp(self, qp, reshape=True):
        n_var = qp.get_num_vars()

        # Extract the objective function from the docplex model
        objective_expr = qp._objective
        linear_constraints = qp.linear_constraints

        # Store variable bounds without converting to binary
        variable_bounds = QAOACVAngleOptimizer.prepare_model(qp)

        # cost operator
        # Create simple cost operator with Ry gates (one for each variable)
        cost = QuantumCircuit(n_var)
        for i in range(n_var):
            # Rx gate
            param_rx = Parameter(f"γ_rx_{i}")
            cost.rx(param_rx, i)

            # Ry gate
            param_ry = Parameter(f"γ_ry_{i}")
            cost.ry(param_ry, i)

            # Rz gate
            param_rz = Parameter(f"γ_rz_{i}")
            cost.rz(param_rz, i)

        # The cost operator always has parameters (one per Ry gate)
        cost_op_has_no_parameter = False

        mixer = self.create_mixer(cost.num_qubits, use_params=cost_op_has_no_parameter)

        # Create initial state: encode continuous features using Ry rotations
        # This is applied once before the QAOA repetitions
        initial_state = QuantumCircuit(n_var)
        continuous_input_params = []
        for i in range(n_var):
            param_input = Parameter(f"θ_{i}")
            continuous_input_params.append(param_input)
            initial_state.ry(param_input, i)

        # QAOA circuit without measurement for state vector
        ansatz_0 = QAOAAnsatz(
            cost_operator=cost,
            reps=self.n_reps,
            initial_state=initial_state,
            mixer_operator=mixer,
        ).decompose()

        def prob(state_vec, i):
            """Extract variable value from state vector using Bloch sphere.

            Extracts the continuous variable value from the Bloch sphere
            Z-component of the reduced density matrix for the i-th qubit.

            Parameters
            ----------
            state_vec : Statevector
                The quantum state vector.
            i : int
                The index of the variable.

            Returns
            -------
            float
                The variable value extracted from the Bloch sphere Z-component.
            """

            # Calculate reduced density matrix for the i-th qubit
            # Trace out all other qubits
            qubits_to_trace = [j for j in range(n_var) if j != i]

            if qubits_to_trace:
                reduced_dm = partial_trace(state_vec, qubits_to_trace)
            else:
                # If only one qubit, use the full density matrix
                reduced_dm = state_vec.to_operator()

            # Create Pauli Z matrix
            pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

            # Calculate expectation value directly
            dm_matrix = reduced_dm.data
            bloch_z = np.real(np.trace(dm_matrix @ pauli_z))

            # Map from Bloch sphere Z component [-1, 1] to variable bounds [lb, ub]
            # bloch_z = 1 corresponds to |0⟩, bloch_z = -1 corresponds to |1⟩
            lb, ub = variable_bounds[i]

            # Map -bloch_z from [-1, 1] to [0, 1]
            normalized = (-bloch_z + 1.0) / 2.0

            # Scale to variable bounds
            value = lb + normalized * (ub - lb)

            return value

        # defining loss function
        self.x_ = []
        self.y_ = []

        def loss(params):
            # Create state vector with current parameters
            circuit = ansatz_0.assign_parameters(params)
            state_vec = Statevector(circuit)

            # Extract variable values from state vector angles
            var_hat = [prob(state_vec, i) for i in range(n_var)]

            cost = objective_expr.evaluate(var_hat)

            # Add penalty for constraint violations
            penalty = 0
            for constraint in linear_constraints:
                value = constraint.linear.evaluate(var_hat)
                violation = (value - constraint.rhs) ** 2  # For EQ constraint
                penalty += 10.0 * violation

            cost_total = cost + penalty

            self.x_.append(len(self.x_))
            self.y_.append(cost_total)
            return cost_total

        # Initial guess for the parameters.
        num_params = ansatz_0.num_parameters
        initial_guess = np.linspace(0, np.pi / 2, num_params)

        bounds = [(0, np.pi / 2)] * num_params

        # minimize function to search for the optimal parameters
        start_time = time.time()
        result = self.optimizer.minimize(loss, initial_guess, bounds=bounds)

        stop_time = time.time()
        self.run_time_ = stop_time - start_time

        self.optim_params_ = result.x

        # running QAOA circuit with optimal parameters
        optimized_circuit = ansatz_0.assign_parameters(self.optim_params_)
        self.state_vector_ = Statevector(optimized_circuit)

        solution = np.array([prob(self.state_vector_, i) for i in range(n_var)])
        self.minimum_ = objective_expr.evaluate(solution)

        if reshape:
            n_channels = int(math.sqrt(solution.shape[0]))
            return np.reshape(solution, (n_channels, n_channels))

        return solution


class QAOACVOptimizer(pyQiskitOptimizer):

    """QAOA with continuous variables.

    Parameters
    ----------
    create_mixer : Callable[int, QuantumCircuit], \
            default=create_mixer_rotational_X_gates(0)
        A delegate that takes into input an angle and returns a QuantumCircuit.
    n_reps : int, default=3
        The number of repetitions for the QAOA ansatz.
        It defines how many time Mixer and Cost operators will be repeated
        in the circuit.
    quantum_instance : QuantumInstance, default=None
        A quantum backend instance.
        If None, AerSimulator will be used.
    optimizer : SciPyOptimizer, default=SPSA()
        An instance of a scipy optimizer to find the optimal weights for the
        parametric circuit (ansatz).

    Notes
    -----
    .. versionadded:: 0.4.0

    Attributes
    ----------
    x_: list[int]
        Indices of the loss function.
    y_: list[int]
        Cost computed by the loss function.
    run_time_: float
        Time taken by the optimizer
    optim_params_: list[float]
        The optimal rotation for the gate in the circuit.
    minimum_: float
        The value of the objective function with the optimal parameters.
    state_vector_: StateVector
        State vector of the optimized quantum circuit
        (optimal parameters assigned to the parametric gates).

    See Also
    --------
    pyQiskitOptimizer
    create_mixer_rotational_X_gates
    """

    def __init__(
        self,
        create_mixer=create_mixer_rotational_X_gates(0),
        n_reps=3,
        quantum_instance=None,
        optimizer=SPSA(),
    ):
        self.n_reps = n_reps
        self.create_mixer = create_mixer
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer

    @staticmethod
    def prepare_model(qp):
        scalers = []
        for v in qp.variables:
            if v.vartype == VarType.CONTINUOUS:
                scaler = MinMaxScaler().fit(
                    np.array([v.lowerbound, v.upperbound]).reshape(-1, 1)
                )
                # print(scaler.data_min_, scaler.data_max_)
                scalers.append(scaler)
                v.vartype = VarType.BINARY
                v.lowerbound = 0
                v.upperbound = 1
        conv = LinearEqualityToPenalty()
        qp = conv.convert(qp)

        return qp, scalers

    def spdmat_var(self, prob, channels, name):
        """ Create docplex matrix variable

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_.
        channels : list
            The list of channels. A channel can be any Python object,
            such as channels'name or number but None.
        name : string
            A custom name for the variable. The name is used internally by docplex
            and may appear if your print the model to a file for example.

        Returns
        -------
        docplex_spdmat : dict
            A docplex representation of a SPD matrix with continuous variables.

        See Also
        -----
        square_cont_mat_var

        Notes
        -----
        .. versionadded:: 0.4.0
        .. versionchanged:: 0.4.0
            rename covmat_var to spdmat_var

        References
        ----------
        .. [1] \
            http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

        """
        return ClassicalOptimizer.spdmat_var(self, prob, channels, name)

    def get_weights(self, prob, classes):
        """Get weights variable

        Helper to create a docplex representation of a weight vector.

        Parameters
        ----------
        prob : Model
            An instance of the docplex model [1]_
        classes : list
            The classes.

        Returns
        -------
        docplex_weights : dict
            A vector of integer decision variables representing weights.

        Notes
        -----
        .. versionadded:: 0.4.0

        """
        return ClassicalOptimizer.get_weights(self, prob, classes)

    def _solve_qp(self, qp, reshape=True):
        quantum_instance = _get_quantum_instance(self)

        n_var = qp.get_num_vars()
        # Extract the objective function from the docplex model
        # We want the object expression with continuous variable
        objective_expr = qp._objective

        # Convert continuous variable to binary ones
        # Get scalers corresponding to the definition range of each variables
        qp, scalers = QAOACVOptimizer.prepare_model(qp)

        # Check all variables are converted to binary, and scalers are registered
        # print(qp.prettyprint(), scalers)

        # cost operator
        # Get operator associated with model
        cost, _offset = qp.to_ising()

        # If the cost operator is a Pauli identity
        # or the cost operator has no parameters
        # the number of parameters in the QAOAAnsatz will be 0.
        # We will then create a mixer with parameters
        # So we get some parameters in the circuit to optimize
        cost_op_has_no_parameter = is_pauli_identity(cost) or len(cost.parameters) == 0

        mixer = self.create_mixer(cost.num_qubits, use_params=cost_op_has_no_parameter)

        # QAOA cirtcuit
        ansatz_0 = QAOAAnsatz(
            cost_operator=cost,
            reps=self.n_reps,
            initial_state=None,
            mixer_operator=mixer,
        ).decompose()
        ansatz = QAOAAnsatz(
            cost_operator=cost,
            reps=self.n_reps,
            initial_state=None,
            mixer_operator=mixer,
        ).decompose()
        ansatz.measure_all()

        def prob(job, i):
            quasi_dists = job.result().quasi_dists[0]
            p = 0
            for key in quasi_dists:
                if key & 2 ** (n_var - 1 - i):
                    p += quasi_dists[key]

            # p is in the range [0, 1].
            # We now need to scale it in the definition
            # range of the continuous variables
            p = scalers[i].inverse_transform([[p]])[0][0]
            return p

        # defining loss function
        self.x_ = []
        self.y_ = []

        def loss(params):
            job = quantum_instance.run(ansatz, params)
            var_hat = [prob(job, i) for i in range(n_var)]
            cost = objective_expr.evaluate(var_hat)
            self.x_.append(len(self.x_))
            self.y_.append(cost)
            return cost

        # Initial guess for the parameters.
        initial_guess = np.array([1, 1] * self.n_reps)

        # minimize function to search for the optimal parameters
        start_time = time.time()
        result = self.optimizer.minimize(loss, initial_guess, bounds=[])
        stop_time = time.time()
        self.run_time_ = stop_time - start_time

        self.optim_params_ = result.x

        # running QAOA circuit with optimal parameters
        job = quantum_instance.run(ansatz, self.optim_params_)
        solution = np.array([prob(job, i) for i in range(n_var)])
        self.minimum_ = objective_expr.evaluate(solution)

        optimized_circuit = ansatz_0.assign_parameters(self.optim_params_)
        self.state_vector_ = Statevector(optimized_circuit)

        if reshape:
            n_channels = int(math.sqrt(solution.shape[0]))
            return np.reshape(solution, (n_channels, n_channels))

        return solution
