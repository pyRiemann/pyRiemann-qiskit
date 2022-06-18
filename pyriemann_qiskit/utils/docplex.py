from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType


def covmat_cont_var(prob, channels, name='cont_covmat'):
    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    return prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                      name=name, lb=-prob.infinity)

def covmat_int_var(prob, channels, name='int_covmat'):
    IntegerVarType.one_letter_symbol = lambda _: 'I'
    return prob.integer_var_matrix(keys1=channels, keys2=channels,
                                   name=name, lb=-prob.infinity)

def covmat_bin_var(prob, channels, name='bin_covmat'):
    BinaryVarType.one_letter_symbol = lambda _: 'B'
    return prob.binary_var_matrix(keys1=channels, keys2=channels,
                                   name=name, lb=-prob.infinity)