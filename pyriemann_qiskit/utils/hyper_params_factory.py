from qiskit.circuit.library import ZZFeatureMap


def gen_zz_feature_map(reps=2, entanglement='linear'):
    """Return a callable that generate a ZZFeatureMap.
    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.

    Parameters
    ----------
    reps : int
        The number of repeated circuits, has a min. value of 1.
    entanglement : Union[str, List[List[int]], Callable[[int], List[int]]]
        Specifies the entanglement structure.

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap
    """
    return lambda dim: ZZFeatureMap(feature_dimension=dim,
                                    reps=reps,
                                    entanglement=entanglement)
