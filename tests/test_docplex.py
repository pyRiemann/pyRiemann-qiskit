import pytest
from docplex.mp.model import Model
from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
from pyriemann_qiskit.utils import (square_cont_mat_var,
                                    square_int_mat_var,
                                    square_bin_mat_var)


@pytest.mark.parametrize('square_mat_var',
                         [(square_cont_mat_var, ContinuousVarType),
                          (square_int_mat_var, IntegerVarType),
                          (square_bin_mat_var, BinaryVarType)])
def test_get_square_cont_var(square_mat_var):
    n = 4
    channels = range(n)
    prob = Model()
    handle = square_mat_var[0]
    expected_result_type = square_mat_var[1]
    mat = handle(prob, channels, name='test')
    first_element = mat[(0, 0)]
    assert len(mat) == n * n
    assert type(first_element.vartype) is expected_result_type
