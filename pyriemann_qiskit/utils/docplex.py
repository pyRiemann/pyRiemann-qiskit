"""This module contains both classic and quantum optimizers, and some helper
functions. The quantum optimizer allows an optimization problem with
constraints (in the form of docplex model) to be run on a quantum computer.
It is for example suitable for:
- MDM optimization problem;
- computation of matrices mean.
"""
import math
import time

from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
import numpy as np
from pyriemann.utils.covariance import normalize
from pyriemann_qiskit.utils.hyper_params_factory import create_mixer_rotational_X_gates
from pyriemann_qiskit.utils.math import is_pauli_identity
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import BackendSampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SLSQP
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters import IntegerToBinary, LinearEqualityToPenalty
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.problems import VarType
from sklearn.preprocessing import MinMaxScaler

from .quantum_provider import get_simulator


_global_optimizer = [None]


def set_global_optimizer(optimizer):
    """Set the value of the global optimizer

    Parameters
    ----------
    optimizer: pyQiskitOptimizer
      An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Notes
    -----
    .. versionadded:: 0.0.4
    """
    _global_optimizer[0] = optimizer


def get_global_optimizer(default):
    """Get the value of the global optimizer

    Parameters
    ----------
    default: pyQiskitOptimizer
      An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.
      It will be returned by default if the global optimizer is None.

    Returns
    -------
    optimizer : pyQiskitOptimizer
        The global optimizer.

    Notes
    -----
    .. versionadded:: 0.0.4
    """
    return _global_optimizer[0] if _global_optimizer[0] is not None else default


def square_cont_mat_var(prob, channels, name="cont_covmat"):
    """Creates a 2-dimensional dictionary of continuous decision variables,
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
        An custom name for the variable. The name is used internally by docplex
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


def square_int_mat_var(prob, channels, upper_bound=7, name="int_covmat"):
    """Creates a 2-dimensional dictionary of integer decision variables,
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
        An custom name for the variable. The name is used internally by docplex
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


def square_bin_mat_var(prob, channels, name="bin_covmat"):
    """Creates a 2-dimensional dictionary of binary decision variables,
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
        An custom name for the variable. The name is used internally by docplex
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
    for running our docplex model independently of the optimizer type
    (such as classical or quantum optimizer).

    Notes
    -----
    .. versionadded:: 0.0.2
    .. versionchanged:: 0.0.4
    """

    def __init__(self):
        pass

    """Hook to apply some transformation on a covariance matrix.

    Parameters
    ----------
    covmat : ndarray, shape (n_features, n_features)
        The covariance matrix.

    Returns
    -------
    transformed_covmat : ndarray, shape (n_features, n_features)
        A transformation of the covariance matrix.

    Notes
    -----
    .. versionadded:: 0.0.2
    """

    def convert_covmat(self, covmat):
        return covmat

    """Helper to create a docplex representation of a
    covariance matrix variable.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        An custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    docplex_covmat : dict
        A square matrix of decision variables representing
        our covariance matrix.

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """

    def covmat_var(self, prob, channels, name):
        raise NotImplementedError()

    def _solve_qp(self, qp, reshape=True):
        raise NotImplementedError()

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

    def solve(self, prob, reshape=True):
        qp = from_docplex_mp(prob)
        return self._solve_qp(qp, reshape)

    """Helper to create a docplex representation of a
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
        our weights.

    Notes
    -----
    .. versionadded:: 0.0.4

    """

    def get_weights(self, prob, classes):
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

    """Helper to create a docplex representation of a
    covariance matrix variable.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        An custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    docplex_covmat : dict
        A square matrix of continuous decision variables representing
        our covariance matrix.

    See Also
    -----
    square_cont_mat_var

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """

    def covmat_var(self, prob, channels, name):
        return square_cont_mat_var(prob, channels, name)

    def _solve_qp(self, qp, reshape=True):
        result = self.optimizer.solve(qp).x
        if reshape:
            n_channels = int(math.sqrt(result.shape[0]))
            return np.reshape(result, (n_channels, n_channels))
        return result

    """Helper to create a docplex representation of a
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
        A vector of continuous decision variables representing
        our weights.

    Notes
    -----
    .. versionadded:: 0.0.4

    """

    def get_weights(self, prob, classes):
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

    def __init__(self, upper_bound=7, quantum_instance=None, optimizer=SLSQP()):
        pyQiskitOptimizer.__init__(self)
        self.upper_bound = upper_bound
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer

    """Transform all values in the covariance matrix
    to integers.

    Example:
    0.123 -> 1230

    Parameters
    ----------
    covmat : ndarray, shape (n_features, n_features)
        The covariance matrix.

    Returns
    -------
    transformed_covmat : ndarray, shape (n_features, n_features)
        A transformation of the covariance matrix.

    Notes
    -----
    .. versionadded:: 0.0.2

    """

    def convert_covmat(self, covmat):
        corr = normalize(covmat, "corr")
        return np.round(corr * self.upper_bound, 0)

    """Helper to create a docplex representation of a
    covariance matrix variable.

    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        An custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    docplex_covmat : dict
        A square matrix of integer decision variables representing
        our covariance matrix.

    See Also
    -----
    square_int_mat_var

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """

    def covmat_var(self, prob, channels, name):
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
            initial_point=[0.0, 0.0],
            callback=_callback,
        )
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = conv.interpret(qaoa.solve(qubo))
        if reshape:
            n_channels = int(math.sqrt(result.shape[0]))
            return np.reshape(result, (n_channels, n_channels))
        return result

    """Helper to create a docplex representation of a
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
        A vector of integer decision variables representing
        our weights.

    Notes
    -----
    .. versionadded:: 0.0.4

    """

    def get_weights(self, prob, classes):
        w = prob.integer_var_matrix(
            keys1=[1], keys2=classes, name="weight", lb=0, ub=self.upper_bound
        )
        w = np.array([w[key] for key in w])
        return w


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
    optimizer : SciPyOptimizer, default=SLSQP()
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
        optimizer=SLSQP(),
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

    """
    Parameters
    ----------
    prob : Model
        An instance of the docplex model [1]_
    channels : list
        The list of channels. A channel can be any Python object,
        such as channels'name or number but None.
    name : string
        An custom name for the variable. The name is used internally by docplex
        and may appear if your print the model to a file for example.

    Returns
    -------
    docplex_covmat : dict
        A square matrix of continuous decision variables representing
        our covariance matrix.

    See Also
    -----
    square_cont_mat_var

    Notes
    -----
    .. versionadded:: 0.4.0

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """

    def covmat_var(self, prob, channels, name):
        return ClassicalOptimizer.covmat_var(self, prob, channels, name)

    """Helper to create a docplex representation of a
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
        A vector of integer decision variables representing
        our weights.

    Notes
    -----
    .. versionadded:: 0.4.0

    """

    def get_weights(self, prob, classes):
        return ClassicalOptimizer.get_weights(self, prob, classes)

    def _solve_qp(self, qp, reshape=True):
        quantum_instance = _get_quantum_instance(self)

        n_var = qp.get_num_vars()
        # Extract the objective function from the docplex model
        # We want the object expression with continuous variable
        objective_expr = qp._objective

        # Convert continous variable to binary ones
        # Get scalers corresponding to the definition range of each variables
        qp, scalers = QAOACVOptimizer.prepare_model(qp)

        # Check all variables are converted to binary, and scalers are registered
        # print(qp.prettyprint(), scalers)

        # cost operator
        # Get operator associated with model
        cost, _offset = qp.to_ising()

        # If the cost operator is a Pauli identity or the cost operator has no parameters
        # the number of parameters in the QAOAAnsatz will be 0.
        # We will then create a mixer with parameters
        # So we get some parameters in our circuit to optimize
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

        # raise ValueError(f"{len(ansatz.parameters)} {len(cost.parameters)} {len(mixer.parameters)} {_is_pauli_identity(cost)}")
        def prob(job, i):
            quasi_dists = job.result().quasi_dists[0]
            p = 0
            for key in quasi_dists:
                if key & 2 ** (n_var - 1 - i):
                    p += quasi_dists[key]

            # p is in the range [0, 1].
            # We now need to scale it in the definition
            # range of our continuous variables
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
        result = self.optimizer.minimize(loss, initial_guess)
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
