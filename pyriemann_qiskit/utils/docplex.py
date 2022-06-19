from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType


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
