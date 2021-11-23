from qiskit.circuit.library import ZZFeatureMap


def gen_zz_feature_map(reps=2, entanglement='linear'):
    """Return a callable that generate a ZZFeatureMap.
    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits.
        The value should be greater or equal to 1.
    entanglement : Union[str, List[List[int]], Callable[[int], List[int]]]
        Specifies the entanglement structure.
        Entanglement structure can be provided with indices or string.
        Possible string values are:
        full, linear, circular, sca and pairwise.
        Please consult the above link for more details
        on entanglement structure:
        https://qiskit.org/documentation/stubs/qiskit.circuit.library.NLocal.html#qiskit.circuit.library.NLocal

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap
    """
    return lambda dim: ZZFeatureMap(feature_dimension=dim,
                                    reps=reps,
                                    entanglement=entanglement)
