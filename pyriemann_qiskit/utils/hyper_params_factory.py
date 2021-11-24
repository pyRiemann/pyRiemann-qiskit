from qiskit.circuit.library import ZZFeatureMap


def gen_zz_feature_map(reps=2, entanglement='linear'):
    """Return a callable that generate a ZZFeatureMap.
    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.
    entanglement : Union[str, List[List[int]], Callable[[int], List[int]]]
        Specifies the entanglement structure.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular', 'sca' and 'pairwise'.
        Consult [1]_ for more details on entanglement structure.

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap

    References
    ----------
    .. [1] https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.NLocal.html
    """
    if reps < 1:
        raise ValueError("Parameter reps must be superior or equal to 1 (Got %d)" % reps)

    return lambda dim: ZZFeatureMap(feature_dimension=dim,
                                    reps=reps,
                                    entanglement=entanglement)
