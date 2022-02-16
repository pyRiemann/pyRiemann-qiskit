from docplex.mp.vartype import ContinuousVarType
from docplex.mp.model import Model


def test_get_square_cont_var():
    channels = range(4)
    prob = Model()
    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    mat = prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                     name='test', lb=-prob.infinity)
    assert mat is not None

