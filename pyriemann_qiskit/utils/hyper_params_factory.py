from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.components.optimizers import SPSA


def gen_zz_feature_map(reps=2, entanglement='linear'):
    """Return a callable that generate a ZZFeatureMap.
    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.
    entanglement : str | list[list[list[int]]] | \
                   Callable[int, list[list[list[int]]]]
        Specifies the entanglement structure.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular' and 'sca'.
        Consult [1]_ for more details on entanglement structure.

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap

    References
    ----------
    .. [1] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.NLocal.html
    """
    if reps < 1:
        raise ValueError("Parameter reps must be superior \
                          or equal to 1 (Got %d)" % reps)

    return lambda n_features: ZZFeatureMap(feature_dimension=n_features,
                                           reps=reps,
                                           entanglement=entanglement)


def get_spsa(max_trials=40, c=(None, None, None, None, 4.0)):
    """Return an instance of SPSA.
    SPSA [1, 2]_ is an algorithmic method for optimizing systems
    with multiple unknown parameters.
    For more details, see [3] and [4].

    Parameters
    ----------
    max_trials : int (default:40)
        Maximum number of iterations to perform.
    c : tuple[float | None] (default:(None, None, None, None, 4.0))
        The 5 control parameters for SPSA algorithms.
        See [3] for implementation details.
        Auto calibration of SPSA will be skiped if one
        of the parameters is different from None.

    Returns
    -------
    ret : SPSA
        An instance of SPSA

    References
    ----------
    .. [1] Spall, J. C. (2012), “Stochastic Optimization,”
           in Handbook of Computational Statistics:
           Concepts and Methods (2nd ed.)
           (J. Gentle, W. Härdle, and Y. Mori, eds.),
           Springer−Verlag, Heidelberg, Chapter 7, pp. 173–201.
           dx.doi.org/10.1007/978-3-642-21551-3_7

    .. [2] Spall, J. C. (1999), "Stochastic Optimization:
           Stochastic Approximation and Simulated Annealing,"
           in Encyclopedia of Electrical and Electronics Engineering
           (J. G. Webster, ed.),
           Wiley, New York, vol. 20, pp. 529–542

    .. [3] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.aqua.components.optimizers.SPSA.html?highlight=spsa#qiskit.aqua.components.optimizers.SPSA

    .. [4] https://www.jhuapl.edu/SPSA/#Overview
    """
    params = {}
    for i in range(5):
        if c[i] is not None:
            params["c" + str(i)] = c[i]
    if len(params) > 0:
        params["skip_calibration"] = True
    return SPSA(max_trials=max_trials, **params)
