"""Module for classification function."""
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.quantum_instance import logger
from qiskit.aqua.algorithms import QSVM, SklearnSVM, VQC
from qiskit.aqua.utils import get_feature_dimension
from qiskit.providers.ibmq import least_busy
from qiskit.aqua.components.optimizers import SPSA
from datetime import datetime
import logging
logger.level = logging.INFO


class QuanticBase(BaseEstimator, ClassifierMixin):

    """Quantum classification.

    This class implements a SKLearn wrapper around Qiskit library.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:
    - There is no noise on a simulated quantum computer (so results are better)
    - Real quantum computer are quicker than simulator
    - Real quantum computer tasks are assigned to a queue
      before being executed on a back-end

    WARNING: At the moment this implementation only supports binary
    classification (eg. Target vs Non-Target experiment)

    Parameters
    ----------
    target : int
        Label of the target symbol
    qAccountToken : string (default:None)
        If quantum==True and qAccountToken provided,
        the classification task will be running on a IBM quantum backend
    processVector : lambda vector: processedVector (default)
        Additional processing on the input vectors. eg: downsampling
    verbose : bool (default:True)
        If true will output all intermediate results and logs
    quantum : Bool (default:True)
        - If true will run on local or remote backend
        (depending on qAccountToken value).
        - If false, will perform classical computing instead
    **parameters : dictionnary
        This is used by  SKLearn with get_params and set_params method
        in order to create a deepcopy of an instance

    Attributes
    ----------
    classes_ : list
        list of classes.
    verbose : see above
    processVector : see above
    qAccountToken : see above
    target : see above
    quantum : see above
    test_input : Dictionnary
        Contains vectorized test set for target and non-target classes
    training_input : Dictionnary
        Contains vectorized training set for target and non-target classes
    provider : IBMQ Provider
        This service provide a remote quantum computer backend
    backend : Quantum computer or simulator
    feature_dim : int
        Size of the vectorized matrix which is passed to quantum classifier
    new_feature_dim : int
        Feature dimension after proccessed by `processVector` lambda
    prev_fit_params : Dictionnary of data and labels
        Keep in memory data and labels passed to fit method.
        This is used for self-calibration.
    feature_map: ZZFeatureMap
        Transform data into quantum space
    quantum_instance: QuantumInstance (Object)
        Backend with specific parameters (number of shots, etc.)

    See Also
    --------
    QuanticSVM
    QuanticVQC

    """

    def __init__(self, target, qAccountToken=None, quantum=True,
                 processVector=lambda v: v, verbose=True, **parameters):
        self.verbose = verbose
        self.log("Initializing Quantum Classifier")
        self.test_input = {}
        self.set_params(**parameters)
        self.processVector = processVector
        self.qAccountToken = qAccountToken
        self.training_input = {}
        self.target = target
        self.quantum = quantum
        if quantum:
            aqua_globals.random_seed = datetime.now().microsecond
            self.log("seed = ", aqua_globals.random_seed)
            if qAccountToken:
                self.log("Real quantum computation will be performed")
                IBMQ.delete_account()
                IBMQ.save_account(qAccountToken)
                IBMQ.load_account()
                self.log("Getting provider...")
                self.provider = IBMQ.get_provider(hub='ibm-q')
            else:
                self.log("Quantum simulation will be performed")
                self.backend = BasicAer.get_backend('qasm_simulator')
        else:
            self.log("Classical SVM will be performed")

    def log(self, *values):
        if self.verbose:
            print("[QClass] ", *values)

    def vectorize(self, X):
        vector = X.reshape(len(X), self.feature_dim)
        return [self.processVector(x) for x in vector]

    def splitTargetAndNonTarget(self, X, y):
        self.log("""[Warning] Spitting target from non target.
                 Only binary classification is supported.""")
        nbSensor = len(X[0])
        try:
            nbSamples = len(X[0][0])
        except Exception:
            nbSamples = 1
        self.feature_dim = nbSensor * nbSamples
        self.log("Feature dimension = ", self.feature_dim)
        Xta = X[y == self.target]
        Xnt = X[np.logical_not(y == self.target)]
        VectorizedXta = self.vectorize(Xta)
        VectorizedXnt = self.vectorize(Xnt)
        self.new_feature_dim = len(VectorizedXta[0])
        self.log("Feature dimension after vector processing = ",
                 self.new_feature_dim)
        return (VectorizedXta, VectorizedXnt)

    def additionnal_setup(self):
        self.log("There is no additional setup.")

    def fit(self, X, y):
        self.log("Fitting: ", X.shape)
        self.prev_fit_params = {"X": X, "y": y}
        self.classes_ = np.unique(y)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)

        self.training_input["Target"] = VectorizedXta
        self.training_input["NonTarget"] = VectorizedXnt
        self.log(get_feature_dimension(self.training_input))
        feature_dim = get_feature_dimension(self.training_input)
        self.feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2,
                                        entanglement='linear')
        self.additionnal_setup()
        if self.quantum:
            if not hasattr(self, "backend"):
                def filters(device):
                    return (
                        device.configuration().n_qubits >= self.new_feature_dim
                        and not device.configuration().simulator
                        and device.status().operational)
                devices = self.provider.backends(filters=filters)
                try:
                    self.backend = least_busy(devices)
                except Exception:
                    self.log("Devices are all busy. Getting the first one...")
                    self.backend = devices[0]
                self.log("Quantum backend = ", self.backend)
            seed_sim = aqua_globals.random_seed
            seed_trans = aqua_globals.random_seed
            self.quantum_instance = QuantumInstance(self.backend, shots=1024,
                                                    seed_simulator=seed_sim,
                                                    seed_transpiler=seed_trans)
        return self

    def get_params(self, deep=True):
        # Class is re-instanciated for each fold of a cv pipeline.
        # Deep copy of the original instance is insure trough this method
        # and the pending one set_params
        return {
            "target": self.target,
            "qAccountToken": self.qAccountToken,
            "quantum": self.quantum,
            "processVector": self.processVector,
            "verbose": self.verbose,
            "test_input": self.test_input,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def run(self, predict_set=None):
        raise Exception("Run method was not implemented")

    def self_calibration(self):
        X = self.prev_fit_params["X"]
        y = self.prev_fit_params["y"]
        test_per = 0.33
        self.log("Test size = ", test_per, " of previous fitting.")
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_per)
        self.fit(X_train, y_train)
        self.score(X_test, y_test)

    def predict(self, X):
        if(len(self.test_input) == 0):
            self.log("There is no test inputs. Self-calibrating...")
            self.self_calibration()
        result = None
        predict_set = self.vectorize(X)
        self.log("Prediction: ", X.shape)
        result = self.run(predict_set)
        self.log("Prediction finished. Returning predicted labels")
        return result["predicted_labels"]

    def predict_proba(self, X):
        self.log("""[WARNING] SVM prediction probabilities are not available.
                 Results from predict will be used instead.""")
        predicted_labels = self.predict(X)
        ret = [np.array([c == 0, c == 1]) for c in predicted_labels]
        return np.array(ret)

    def score(self, X, y):
        self.log("Scoring: ", X.shape)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)
        self.test_input = {}
        self.test_input["Target"] = VectorizedXta
        self.test_input["NonTarget"] = VectorizedXnt
        result = self.run()
        balanced_accuracy = result["testing_accuracy"]
        self.log("Balanced accuracy = ", balanced_accuracy)
        return balanced_accuracy


class QuanticSVM(QuanticBase):

    """Quantum-enhanced SVM classification.

    This class implements SVC on a quantum machine.
    Note if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    See Also
    --------
    QuanticBase

    """

    def run(self, predict_set=None):
        self.log("SVM classification running...")
        if self.quantum:
            self.log("Quantum instance is ", self.quantum_instance)
            qsvm = QSVM(self.feature_map, self.training_input,
                        self.test_input, predict_set)
            result = qsvm.run(self.quantum_instance)
        else:
            result = SklearnSVM(self.training_input,
                                self.test_input, predict_set).run()
        self.log(result)
        return result


class QuanticVQC(QuanticBase):

    """Variational Quantum Classifier

    Note there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not)

    Parameters
    ----------
    target : see QuanticBase
    qAccountToken : see QuanticBase
    processVector : see QuanticBase
    verbose : see QuanticBase
    parameters : see QuanticBase

    Attributes
    ----------
    optimizer: SPSA
        SPSA is a descent method capable of finding global minima
        https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.SPSA.html
    var_form: TwoLocal
        In quantum mechanics, the variational method is one way of finding
        approximations to the lowest energy eigenstate
        https://qiskit.org/documentation/apidoc/qiskit.aqua.components.variational_forms.html

    See Also
    --------
    QuanticBase

    """

    def __init__(self, target, qAccountToken=None,
                 processVector=lambda v: v, verbose=True, **parameters):
        QuanticBase.__init__(self, target=target, qAccountToken=qAccountToken,
                             processVector=processVector, verbose=verbose,
                             **parameters)

    def additionnal_setup(self):
        self.optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
        self.var_form = TwoLocal(self.new_feature_dim,
                                 ['ry', 'rz'], 'cz', reps=3)

    def run(self, predict_set=None):
        self.log("VQC classification running...")
        vqc = VQC(self.optimizer, self.feature_map, self.var_form,
                  self.training_input, self.test_input, predict_set)
        result = vqc.run(self.quantum_instance)
        return result
