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

    * There is no noise on a simulated quantum computer (so results are better)
    * Real quantum computer are quicker than simulator
    * Real quantum computer tasks are assigned to a queue
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
    verbose_ : see above
    _classes : list
        list of classes.
    _processVector : see above
    _qAccountToken : see above
    _target : see above
    _quantum : see above
    _test_input : Dictionnary
        Contains vectorized test set for target and non-target classes
    _training_input : Dictionnary
        Contains vectorized training set for target and non-target classes
    _provider : IBMQ Provider
        This service provide a remote quantum computer backend
    _backend : Quantum computer or simulator
    _feature_dim : int
        Size of the vectorized matrix which is passed to quantum classifier
    _new_feature_dim : int
        Feature dimension after proccessed by `processVector` lambda
    _prev_fit_params : Dictionnary of data and labels
        Keep in memory data and labels passed to fit method.
        This is used for self-calibration.
    _feature_map: ZZFeatureMap
        Transform data into quantum space
    _quantum_instance: QuantumInstance (Object)
        Backend with specific parameters (number of shots, etc.)

    See Also
    --------
    QuanticSVM
    QuanticVQC

    """

    def __init__(self, target, qAccountToken=None, quantum=True,
                 processVector=lambda v: v, verbose=True, **parameters):
        self.verbose_ = verbose
        self._log("Initializing Quantum Classifier")
        self._test_input = {}
        self.set_params(**parameters)
        self._processVector = processVector
        self._qAccountToken = qAccountToken
        self._training_input = {}
        self._target = target
        self._quantum = quantum
        if quantum:
            aqua_globals.random_seed = datetime.now().microsecond
            self._log("seed = ", aqua_globals.random_seed)
            if qAccountToken:
                self._log("Real quantum computation will be performed")
                IBMQ.delete_account()
                IBMQ.save_account(qAccountToken)
                IBMQ.load_account()
                self._log("Getting provider...")
                self._provider = IBMQ.get_provider(hub='ibm-q')
            else:
                self._log("Quantum simulation will be performed")
                self._backend = BasicAer.get_backend('qasm_simulator')
        else:
            self._log("Classical SVM will be performed")

    def _log(self, *values):
        if self.verbose_:
            print("[QClass] ", *values)

    def _vectorize(self, X):
        vector = X.reshape(len(X), self._feature_dim)
        return [self._processVector(x) for x in vector]

    def _split_target_and_non_target(self, X, y):
        self._log("""[Warning] Spitting target from non target.
                 Only binary classification is supported.""")
        nbSensor = len(X[0])
        try:
            nbSamples = len(X[0][0])
        except Exception:
            nbSamples = 1
        self._feature_dim = nbSensor * nbSamples
        self._log("Feature dimension = ", self._feature_dim)
        Xta = X[y == self._target]
        Xnt = X[np.logical_not(y == self._target)]
        VectorizedXta = self._vectorize(Xta)
        VectorizedXnt = self._vectorize(Xnt)
        self._new_feature_dim = len(VectorizedXta[0])
        self._log("Feature dimension after vector processing = ",
                 self._new_feature_dim)
        return (VectorizedXta, VectorizedXnt)

    def _additional_setup(self):
        self._log("There is no additional setup.")

    def fit(self, X, y):
        self._log("Fitting: ", X.shape)
        self._prev_fit_params = {"X": X, "y": y}
        self._classes = np.unique(y)
        VectorizedXta, VectorizedXnt = self._split_target_and_non_target(X, y)

        self._training_input["Target"] = VectorizedXta
        self._training_input["NonTarget"] = VectorizedXnt
        self._log(get_feature_dimension(self._training_input))
        feature_dim = get_feature_dimension(self._training_input)
        self._feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2,
                                        entanglement='linear')
        self._additional_setup()
        if self._quantum:
            if not hasattr(self, "_backend"):
                def filters(device):
                    return (
                        device.configuration().n_qubits >= self._new_feature_dim
                        and not device.configuration().simulator
                        and device.status().operational)
                devices = self._provider.backends(filters=filters)
                try:
                    self._backend = least_busy(devices)
                except Exception:
                    self._log("Devices are all busy. Getting the first one...")
                    self._backend = devices[0]
                self._log("Quantum backend = ", self._backend)
            seed_sim = aqua_globals.random_seed
            seed_trans = aqua_globals.random_seed
            self._quantum_instance = QuantumInstance(self._backend, shots=1024,
                                                    seed_simulator=seed_sim,
                                                    seed_transpiler=seed_trans)
        return self

    def get_params(self, deep=True):
        # Class is re-instanciated for each fold of a cv pipeline.
        # Deep copy of the original instance is insure trough this method
        # and the pending one set_params
        return {
            "target": self._target,
            "qAccountToken": self._qAccountToken,
            "quantum": self._quantum,
            "processVector": self._processVector,
            "verbose": self.verbose_,
            "test_input": self._test_input,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _run(self, predict_set=None):
        raise Exception("Run method was not implemented")

    def _self_calibration(self):
        X = self._prev_fit_params["X"]
        y = self._prev_fit_params["y"]
        test_per = 0.33
        self._log("Test size = ", test_per, " of previous fitting.")
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_per)
        self.fit(X_train, y_train)
        self.score(X_test, y_test)

    def predict(self, X):
        if(len(self._test_input) == 0):
            self._log("There is no test inputs. Self-calibrating...")
            self._self_calibration()
        result = None
        predict_set = self._vectorize(X)
        self._log("Prediction: ", X.shape)
        result = self._run(predict_set)
        self._log("Prediction finished. Returning predicted labels")
        return result["predicted_labels"]

    def predict_proba(self, X):
        self._log("""[WARNING] SVM prediction probabilities are not available.
                 Results from predict will be used instead.""")
        predicted_labels = self.predict(X)
        ret = [np.array([c == 0, c == 1]) for c in predicted_labels]
        return np.array(ret)

    def score(self, X, y):
        self._log("Scoring: ", X.shape)
        VectorizedXta, VectorizedXnt = self._split_target_and_non_target(X, y)
        self._test_input = {}
        self._test_input["Target"] = VectorizedXta
        self._test_input["NonTarget"] = VectorizedXnt
        result = self._run()
        balanced_accuracy = result["testing_accuracy"]
        self._log("Balanced accuracy = ", balanced_accuracy)
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

    def _run(self, predict_set=None):
        self._log("SVM classification running...")
        if self._quantum:
            self._log("Quantum instance is ", self._quantum_instance)
            qsvm = QSVM(self._feature_map, self._training_input,
                        self._test_input, predict_set)
            result = qsvm.run(self._quantum_instance)
        else:
            result = SklearnSVM(self._training_input,
                                self._test_input, predict_set).run()
        self._log(result)
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

    def _additional_setup(self):
        self.optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
        self.var_form = TwoLocal(self._new_feature_dim,
                                 ['ry', 'rz'], 'cz', reps=3)

    def _run(self, predict_set=None):
        self._log("VQC classification running...")
        vqc = VQC(self.optimizer, self._feature_map, self.var_form,
                  self._training_input, self._test_input, predict_set)
        result = vqc.run(self._quantum_instance)
        return result
