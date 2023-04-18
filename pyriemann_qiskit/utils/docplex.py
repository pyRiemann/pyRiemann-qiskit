"""This module contains both classic and quantum optimizers, and some helper
functions. The quantum optimizer allows an optimization problem with
constraints (in the form of docplex model) to be run on a quantum computer.
It is for example suitable for:
- MDM optimization problem;
- computation of matrices mean.
"""
import math
import numpy as np
from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import (CobylaOptimizer,
                                            MinimumEigenOptimizer)
from qiskit_optimization.converters import IntegerToBinary
from qiskit_optimization.translators import from_docplex_mp
from pyriemann_qiskit.utils import cov_to_corr_matrix


def square_cont_mat_var(prob, channels,
                        name='cont_covmat'):
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
    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    return prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                      name=name, lb=-prob.infinity)


def square_int_mat_var(prob, channels, upper_bound=7,
                       name='int_covmat'):
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
    IntegerVarType.one_letter_symbol = lambda _: 'I'
    return prob.integer_var_matrix(keys1=channels, keys2=channels,
                                   name=name, lb=0, ub=upper_bound)


def square_bin_mat_var(prob, channels,
                       name='bin_covmat'):
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
    BinaryVarType.one_letter_symbol = lambda _: 'B'
    return prob.binary_var_matrix(keys1=channels, keys2=channels,
                                  name=name)


class pyQiskitOptimizer():

    """Wrapper for Qiskit optimizer.

    This class is an abstract class which provides an interface
    for running our docplex model independently of the optimizer type
    (such as classical or quantum optimizer).

    Notes
    -----
    .. versionadded:: 0.0.2
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

    def _solve_qp(self, qp):
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

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """
    def solve(self, prob):
        qp = from_docplex_mp(prob)
        return self._solve_qp(qp)


class ClassicalOptimizer(pyQiskitOptimizer):

    """Wrapper for the classical Cobyla optimizer.

    Notes
    -----
    .. versionadded:: 0.0.2

    See Also
    --------
    pyQiskitOptimizer

    """
    def __init__(self):
        pyQiskitOptimizer.__init__(self)

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

    def _solve_qp(self, qp):
        result = CobylaOptimizer(rhobeg=0.01, rhoend=0.0001).solve(qp).x
        n_channels = int(math.sqrt(result.shape[0]))
        return np.reshape(result, (n_channels, n_channels))


class NaiveQAOAOptimizer(pyQiskitOptimizer):

    """Wrapper for the quantum optimizer QAOA.

    Attributes
    ----------
    upper_bound : int (default: 7)
        The maximum integer value for matrix normalization.

    Notes
    -----
    .. versionadded:: 0.0.2

    See Also
    --------
    pyQiskitOptimizer
    """
    def __init__(self, upper_bound=7):
        pyQiskitOptimizer.__init__(self)
        self.upper_bound = upper_bound

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
        corr = cov_to_corr_matrix(covmat)
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

    def _solve_qp(self, qp):
        conv = IntegerToBinary()
        qubo = conv.convert(qp)
        backend = Aer.get_backend('aer_simulator')
        quantum_instance = QuantumInstance(backend)
        qaoa_mes = QAOA(quantum_instance=quantum_instance,
                        initial_point=[0., 0.])
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = conv.interpret(qaoa.solve(qubo))
        n_channels = int(math.sqrt(result.shape[0]))
        return np.reshape(result, (n_channels, n_channels))
