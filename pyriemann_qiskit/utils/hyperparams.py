from qiskit.circuit.library import ZZFeatureMap

default = {
    "gamma": None, #quantum=False
    "lambda2":0.001, #QSVC
    "feature_map":
        lambda dim: 
            ZZFeatureMap(feature_dimension=dim, reps=2, entanglement='linear'),
    "nshots":None,
    "optimizer":None, #VQC
    "var_form":None, #VQC
    "enforce_spd":None,
    "output_norm":None, #quantum=Fale
    "l2reg":None
}


class QuanticHyperParams():
    """
    This class is a wrapper around all hyper parameters for quantum classifier.

    Parameters
    ----------
    gamma: TODO
    lambda2 : L2 norm regularization factor (QSVM)
    feature_map : (feature_dim)->(Union[QuantumCircuit, FeatureMap])
        Feature map module, used to transform data (QSVM and VQC)

    Notes
    -----
    .. versionadded:: 0.0.1


    See Also
    --------
    QuanticClassifierBase
    QuanticSVM
    QuanticVQC

    """
    def __init__(self, gamma=None, lambda2=default["lambda2"], feature_map=default["feature_map"]):
        self.gamma = gamma
        self.lambda2 = lambda2
        self.feature_map = feature_map
        