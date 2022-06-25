from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters import IntegerToBinary
from qiskit_optimization.problems import QuadraticProgram
import numpy as np

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

    Returns
    -------
    square_mat : dict
        A square matrix of continuous decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    return prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                      name=name, lb=-prob.infinity)


def square_int_mat_var(prob, channels,
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

    Returns
    -------
    square_mat : dict
        A square matrix of integer decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    IntegerVarType.one_letter_symbol = lambda _: 'I'
    return prob.integer_var_matrix(keys1=channels, keys2=channels,
                                   name=name, lb=-prob.infinity)


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

    Returns
    -------
    square_mat : dict
        A square matrix of binary decision variables.
        Access element (i, j) with square_mat[(i, j)].
        Indices start with 0.

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model
    """
    BinaryVarType.one_letter_symbol = lambda _: 'B'
    return prob.binary_var_matrix(keys1=channels, keys2=channels,
                                  name=name)


class pyQiskitOptimizer():
    def convert_covmat(self, covmat, precision=None):
        return covmat

    def covmat_var(self, prob, channels, name):
        raise NotImplementedError()

    def _solve_qp(self, qp):
        raise NotImplementedError()

    def solve(self, prob):
        qp = QuadraticProgram()
        qp.from_docplex(prob)
        return self._solve_qp(qp)


class ClassicalOptimizer(pyQiskitOptimizer):
    def covmat_var(self, prob, channels, name):
        return square_cont_mat_var(prob, channels)

    def _solve_qp(self, qp):
        return CobylaOptimizer(rhobeg=0.01, rhoend=0.0001).solve(qp)


class NaiveQAOAOptimizer(pyQiskitOptimizer):
    def convert_covmat(self, covmat, precision=10**4):
        return np.round(covmat * precision, 0)

    def covmat_var(self, prob, channels, name):
        return square_int_mat_var(prob, channels)

    def _solve_qp(self, qp):
        conv = IntegerToBinary()
        qubo = conv.convert(qp)
        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend)
        qaoa_mes = QAOA(quantum_instance=quantum_instance,
                        initial_point=[0., 0.])
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        return qaoa.solve(qubo)
