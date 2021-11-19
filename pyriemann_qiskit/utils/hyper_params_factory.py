from qiskit.circuit.library import ZZFeatureMap


def gen_zz_feature_map(reps=2, entanglement='linear'):
    return lambda dim: ZZFeatureMap(feature_dimension=dim,
                                    reps=reps,
                                    entanglement=entanglement)
