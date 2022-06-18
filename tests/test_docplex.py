from pyriemann_qiskit.utils.docplex import covmat_cont_var, covmat_int_var, covmat_bin_var
from docplex.mp.model import Model


def test_get_square_cont_var():
    channels = range(4)
    prob = Model()
    mat = covmat_cont_var(prob, channels, name='test')
    assert mat is not None
