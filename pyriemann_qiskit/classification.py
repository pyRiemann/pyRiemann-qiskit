"""Module for classification function."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import TwoLocal
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.quantum_instance import logger
from qiskit.aqua.algorithms import QSVM, SklearnSVM, VQC
from qiskit.aqua.utils import get_feature_dimension
from qiskit.providers.ibmq import least_busy
from qiskit.aqua.components.optimizers import SPSA
from datetime import datetime
import logging
from .utils.hyper_params_factory import gen_zz_feature_map

logger.level = logging.INFO


class QuanticClassifierBase(BaseEstimator, ClassifierMixin):

    """Quantum classification.

    This class implements a SKLearn wrapper around Qiskit library [1]_.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:

    * there is no noise on a simulated quantum computer
      (so results are better);
    * real quantum computer are quicker than simulator;
    * real quantum computer tasks are assigned to a queue
      before being executed on a back-end.

    WARNING: At the moment this implementation only supports binary
    classification.

    Parameters
    ----------
    quantum : bool (default: True)
        - If true will run on local or remote backend
        (depending on q_account_token value).
        - If false, will perform classical computing instead
    q_account_token : string (default:None)
        If quantum==True and q_account_token provided,
        the classification task will be running on a IBM quantum backend
    verbose : bool (default:True)
        If true will output all intermediate results and logs
    shots : int (default:1024)
        Number of repetitions of each circuit, for sampling
    gen_feature_map : Callable[int, QuantumCircuit | FeatureMap]
        Function generating a feature map to encode data into a quantum state.

    Notes
    -----
    .. versionadded:: 0.0.1

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    QuanticSVM
    QuanticVQC

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    """

    def __init__(self, quantum=True, q_account_token=None, verbose=True,
                 shots=1024, gen_feature_map=gen_zz_feature_map()):
        self.verbose = verbose
        self._log("Initializing Quantum Classifier")
        self.q_account_token = q_account_token
        self.quantum = quantum
        self.shots = shots
        self.gen_feature_map = gen_feature_map
        # protected field for child classes
        self._training_input = {}

    def _init_quantum(self):
        if self.quantum:
            aqua_globals.random_seed = datetime.now().microsecond
            self._log("seed = ", aqua_globals.random_seed)
            if self.q_account_token:
                self._log("Real quantum computation will be performed")
                IBMQ.delete_account()
                IBMQ.save_account(self.q_account_token)
                IBMQ.load_account()
                self._log("Getting provider...")
                self._provider = IBMQ.get_provider(hub='ibm-q')
            else:
                self._log("Quantum simulation will be performed")
                self._backend = BasicAer.get_backend('qasm_simulator')
        else:
            self._log("Classical SVM will be performed")

    def _log(self, *values):
        if self.verbose:
            print("[QClass] ", *values)

    def _split_classes(self, X, y):
        self._log("[Warning] Splitting first class from second class."
                  "Only binary classification is supported.")
        X_class1 = X[y == self.classes_[1]]
        X_class0 = X[y == self.classes_[0]]
        return (X_class1, X_class0)

    def _map_classes_to_0_1(self, y):
        y_copy = y.copy()
        y_copy[y == self.classes_[0]] = 0
        y_copy[y == self.classes_[1]] = 1
        return y_copy

    def _map_0_1_to_classes(self, y):
        y_copy = y.copy()
        y_copy[y == 0] = self.classes_[0]
        y_copy[y == 1] = self.classes_[1]
        return y_copy

    def fit(self, X, y):
        """Get a quantum backend and fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : QuanticClassifierBase instance
            The QuanticClassifierBase instance.
        """
        self._init_quantum()

        self._log("Fitting: ", X.shape)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise Exception("Only binary classification \
                             is currently supported.")
        y = self._map_classes_to_0_1(y)

        class1, class0 = self._split_classes(X, y)

        self._training_input[self.classes_[1]] = class1
        self._training_input[self.classes_[0]] = class0

        feature_dim = get_feature_dimension(self._training_input)
        self._log("Feature dimension = ", feature_dim)
        self._feature_map = self.gen_feature_map(feature_dim)
        if self.quantum:
            if not hasattr(self, "_backend"):
                def filters(device):
                    return (
                      device.configuration().n_qubits >= feature_dim
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
            seed_trs = aqua_globals.random_seed
            self._quantum_instance = QuantumInstance(self._backend,
                                                     shots=self.shots,
                                                     seed_simulator=seed_sim,
                                                     seed_transpiler=seed_trs)
        self._classifier = self._init_algo(feature_dim)
        self._train(X, y)
        return self

    def _init_algo(self, feature_dim):
        raise Exception("Init algo method was not implemented")

    def _train(self, X, y):
        self._log("Training...")
        if self.quantum:
            self._classifier.train(X, y, self._quantum_instance)
        else:
            self._classifier.train(X, y)

    def score(self, X, y):
        """Return the testing accuracy.
           You might want to use a different metric by using sklearn
           cross_val_score
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        accuracy : double
            Accuracy of predictions from X with respect y.
        """
        y = self._map_classes_to_0_1(y)
        self._log("Testing...")
        if self.quantum:
            return self._classifier.test(X, y, self._quantum_instance)
        else:
            return self._classifier.test(X, y)

    def _predict(self, X):
        self._log("Prediction: ", X.shape)
        print(self._training_input)
        result = self._classifier.predict(X)
        self._log("Prediction finished.")
        return result


class QuanticSVM(QuanticClassifierBase):

    """Quantum-enhanced SVM classification.

    This class implements SVC [1]_ on a quantum machine [2]_.
    Note if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    QuanticClassifierBase

    References
    ----------
    .. [1] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    .. [2] V. Havlíček et al.,
           ‘Supervised learning with quantum-enhanced feature spaces’,
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    """

    def _init_algo(self, feature_dim):
        # Although we do not train the classifier at this location
        # training_input are required by Qiskit library.
        self._log("SVM initiating algorithm")
        if self.quantum:
            classifier = QSVM(self._feature_map, self._training_input)
        else:
            classifier = SklearnSVM(self._training_input)
        return classifier

    def predict_proba(self, X):
        """This method is implemented for compatibility purpose
           as SVM prediction probabilities are not available.
           This method assigns to each trial a boolean which value
           depends on wheter the label was assigned to classes 0 or 1

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, 0] == True if the nth sample is assigned to 1st class;
            prob[n, 1] == True if the nth sample is assigned to 2nd class.
        """
        predicted_labels = self.predict(X)
        ret = [np.array([c == self.classes_[0], c == self.classes_[1]])
               for c in predicted_labels]
        return np.array(ret)

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        labels = self._predict(X)
        return self._map_0_1_to_classes(labels)


class QuanticVQC(QuanticClassifierBase):

    """Variational Quantum Classifier

    Note there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not)

    Parameters
    ----------
    q_account_token : string (default:None)
        If quantum==True and q_account_token provided,
        the classification task will be running on a IBM quantum backend
    verbose : bool (default:True)
        If true will output all intermediate results and logs

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    QuanticClassifierBase

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    .. [2] V. Havlíček et al.,
           ‘Supervised learning with quantum-enhanced feature spaces’,
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    """

    def __init__(self, q_account_token=None,
                 verbose=True, **parameters):
        QuanticClassifierBase.__init__(self,
                                       q_account_token=q_account_token,
                                       verbose=verbose)

    def _init_algo(self, feature_dim):
        self._log("VQC training...")
        self._optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
        self._var_form = TwoLocal(feature_dim,
                                  ['ry', 'rz'], 'cz', reps=3)
        # Although we do not train the classifier at this location
        # training_input are required by Qiskit library.
        vqc = VQC(self._optimizer, self._feature_map, self._var_form,
                  self._training_input)
        return vqc

    def predict_proba(self, X):
        """This method is implemented for compatibility purpose
           as SVM prediction probabilities are not available.
           This method assigns to each trial a boolean which value
           depends on wheter the label was assigned to classes 0 or 1

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, 0] == True if the nth sample is assigned to 1st class;
            prob[n, 1] == True if the nth sample is assigned to 2nd class.
        """
        proba, _ = self._predict(X)
        return proba

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        _, labels = self._predict(X)
        return self._map_0_1_to_classes(labels)
