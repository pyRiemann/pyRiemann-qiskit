import pytest
from docplex.mp.model import Model
from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
from pyriemann_qiskit.utils import (
    square_cont_mat_var,
    square_int_mat_var,
    square_bin_mat_var,
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
)


@pytest.mark.parametrize(
    "square_mat_var",
    [
        (square_cont_mat_var, ContinuousVarType),
        (square_int_mat_var, IntegerVarType),
        (square_bin_mat_var, BinaryVarType),
    ],
)
def test_get_square_cont_var(square_mat_var):
    """
    Tests that:
      - square_cont_mat_var is returning a variable of type ContinuousVarType
      - square_cont_mat_var is returning a variable of type IntegerVarType
      - square_bin_mat_var is returning a variable of type BinaryVarType
      - all returned variables are square matrices of size 'n' where 'n' is
      the (given) number of channels.
    """
    n = 4
    channels = range(n)
    prob = Model()
    handle = square_mat_var[0]
    expected_result_type = square_mat_var[1]
    mat = handle(prob, channels, name="test")
    first_element = mat[(0, 0)]
    assert len(mat) == n * n
    assert type(first_element.vartype) is expected_result_type


@pytest.mark.parametrize("optimizer", [ClassicalOptimizer, NaiveQAOAOptimizer])
def test_optimizer_creation(optimizer):
    assert optimizer()
