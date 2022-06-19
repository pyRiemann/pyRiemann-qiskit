from pyriemann_qiskit.utils.docplex import square_cont_mat_var
from docplex.mp.model import Model


def test_get_square_cont_var():
    channels = range(4)
    prob = Model()
    mat = square_cont_mat_var(prob, channels, name='test')
    # check len and type
    assert mat is not None
