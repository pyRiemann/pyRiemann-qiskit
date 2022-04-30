"""Module for classification function."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from qiskit import BasicAer, IBMQ
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.quantum_instance import logger
from qiskit.aqua.algorithms import QSVM, SklearnSVM, VQC
from qiskit.aqua.utils import get_feature_dimension
from qiskit.providers.ibmq import least_busy
from datetime import datetime
import logging
from .utils.hyper_params_factory import (gen_zz_feature_map,
                                         gen_two_local,
                                         get_spsa)
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

logger.level = logging.INFO


class QuanticClassifierBase(BaseEstimator, ClassifierMixin):

    """Quantum classification.

    This class implements a SKLearn wrapper around Qiskit library [1]_.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:

    * there is no noise on a simulated quantum computer
      (so results are better)
    * a real quantum computer is quicker than a quantum simulator
    * tasks on a real quantum computer are assigned to a queue
      before being executed on a back-end (delayed execution)

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
    gen_feature_map : Callable[int, QuantumCircuit | FeatureMap] \
                      (default : Callable[int, ZZFeatureMap])
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
        """Uses a quantum backend and fits the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Raises
        ------
        Exception
            Raised if the number of classes is different from 2

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

        n_features = get_feature_dimension(self._training_input)
        self._log("Feature dimension = ", n_features)
        self._feature_map = self.gen_feature_map(n_features)
        if self.quantum:
            if not hasattr(self, "_backend"):
                def filters(device):
                    return (
                      device.configuration().n_qubits >= n_features
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
        self._classifier = self._init_algo(n_features)
        self._train(X, y)
        return self

    def _init_algo(self, n_features):
        raise Exception("Init algo method was not implemented")

    def _train(self, X, y):
        self._log("Training...")
        if self.quantum:
            self._classifier.train(X, y, self._quantum_instance)
        else:
            self._classifier.train(X, y)

    def score(self, X, y):
        """Returns the testing accuracy.
           You might want to use a different metric by using sklearn
           cross_val_score

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y: ndarray, shape (n_samples,)
            Predicted target vector relative to X.

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
    Note that if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    Notes
    -----
    .. versionadded:: 0.0.1

    Parameters
    ----------
    gamma : float | None (default:None)
        Used as input for sklearn rbf_kernel which is used internally.
        See [3]_ for more information about gamma.

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

    .. [3] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html

    """

    def __init__(self, gamma=None, **parameters):
        QuanticClassifierBase.__init__(self, **parameters)
        self.gamma = gamma

    def _init_algo(self, n_features):
        # Although we do not train the classifier at this location
        # training_input are required by Qiskit library.
        self._log("SVM initiating algorithm")
        if self.quantum:
            classifier = QSVM(self._feature_map, self._training_input)
        else:
            classifier = SklearnSVM(self._training_input, gamma=self.gamma)
        return classifier

    def predict_proba(self, X):
        """This method is implemented for compatibility purpose
           as SVM prediction probabilities are not available.
           This method assigns a boolean value to each trial which
           depends on whether the label was assigned to class 0 or 1

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
        """Calculates the predictions.

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

    Note that there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not).

    Parameters
    ----------
    optimizer : Optimizer (default:SPSA)
        The classical optimizer to use.
        See [3] for details.
    gen_var_form : Callable[int, QuantumCircuit | VariationalForm] \
                   (default: Callable[int, TwoLocal])
        Function generating a variational form instance.

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    QuanticClassifierBase

    Raises
    ------
    ValueError
        Raised if ``quantum`` is False

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    .. [2] V. Havlíček et al.,
           ‘Supervised learning with quantum-enhanced feature spaces’,
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    .. [3] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.aqua.algorithms.VQC.html

    """

    def __init__(self, optimizer=get_spsa(), gen_var_form=gen_two_local(),
                 **parameters):
        if "quantum" in parameters and not parameters["quantum"]:
            raise ValueError("VQC can only run on a quantum \
                              computer or simulator.")
        QuanticClassifierBase.__init__(self, **parameters)
        self.optimizer = optimizer
        self.gen_var_form = gen_var_form

    def _init_algo(self, n_features):
        self._log("VQC training...")
        var_form = self.gen_var_form(n_features)
        # Although we do not train the classifier at this location
        # training_input are required by Qiskit library.
        vqc = VQC(self.optimizer, self._feature_map, var_form,
                  self._training_input)
        return vqc

    def predict_proba(self, X):
        """Returns the probabilities associated with predictions.

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
        """Calculates the predictions.

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


class QuantumClassifierWithDefaultRiemannianPipeline(BaseEstimator,
                                                     ClassifierMixin,
                                                     TransformerMixin):

    """Default pipeline with Riemann Geometry and a quantum classifier.

    Projects the data into the tangent space of the Riemannian manifold
    and applies quantum classification.

    The type of quantum classification (quantum SVM or VQC) depends on
    the value of the parameters.

    Data are entangled using a ZZFeatureMap. A SPSA optimizer and a two-local
    circuits are used in addition when the VQC is selected.



    Parameters
    ----------
    nfilter : int (default: 1)
        The number of filter for the xDawnFilter.
        The number of components selected is 2 x nfilter.
    dim_red : TransformerMixin (default: PCA())
        A transformer that will reduce the dimension of the feature,
        after the data are projected into the tangent space.
    gamma : float | None (default:None)
        Used as input for sklearn rbf_kernel which is used internally.
        See [1]_ for more information about gamma.
    shots : int (default:1024)
        Number of repetitions of each circuit, for sampling.
    feature_entanglement : str | list[list[list[int]]] | \
                   Callable[int, list[list[list[int]]]]
        Specifies the entanglement structure for the ZZFeatureMap.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular' and 'sca'.
        Consult [2]_ for more details on entanglement structure.
    feature_reps : int (default: 2)
        The number of repeated circuits for the ZZFeatureMap,
        greater or equal to 1.
    spsa_trials : int (default: 40)
        Maximum number of iterations to perform using SPSA optimizer.
    two_local_reps : int (default: 3)
        The number of repetition for the two-local cricuit.

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    XdawnCovariances
    TangentSpace
    gen_zz_feature_map
    gen_two_local
    get_spsa
    QuanticVQC
    QuanticSVM

    References
    ----------
    .. [1] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html

    .. [2] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.NLocal.html

    """

    def __init__(self, nfilter=1, dim_red=PCA(),
                 gamma=None, shots=1024, feature_entanglement='full',
                 feature_reps=2, spsa_trials=None, two_local_reps=None,
                 **params):

        self.nfilter = nfilter
        self.dim_red = dim_red
        self.gamma = gamma
        self.shots = shots
        self.feature_entanglement = feature_entanglement
        self.feature_reps = feature_reps
        self.spsa_trials = spsa_trials
        self.two_local_reps = two_local_reps

        is_vqc = spsa_trials and two_local_reps
        is_quantum = shots is not None

        feature_map = gen_zz_feature_map(feature_reps, feature_entanglement)
        # verbose is passed as an additional parameter to quantum classifiers.
        self.verbose = "verbose" in params and params["verbose"]
        if is_vqc:
            self._log("QuanticVQC chosen.")
            clf = QuanticVQC(optimizer=get_spsa(spsa_trials),
                             gen_var_form=gen_two_local(two_local_reps),
                             gen_feature_map=feature_map,
                             shots=self.shots,
                             quantum=is_quantum,
                             **params)
        else:
            self._log("QuanticSVM chosen.")
            clf = QuanticSVM(quantum=is_quantum, gamma=gamma,
                             gen_feature_map=feature_map,
                             shots=shots, **params)

        self._pipe = make_pipeline(XdawnCovariances(nfilter=nfilter),
                                   TangentSpace(), dim_red, clf)

    def _log(self, trace):
        if self.verbose:
            print("[QuantumClassifierWithDefaultRiemannianPipeline] ", trace)

    def fit(self, X, y):
        """Train the riemann quantum classifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : QuantumClassifierWithDefaultRiemannianPipeline instance
            The QuantumClassifierWithDefaultRiemannianPipeline instance
        """

        self.classes_ = np.unique(y)
        self._pipe.fit(X, y)
        return self

    def score(self, X, y):
        """Return the accuracy.
        You might want to use a different metric by using sklearn
        cross_val_score

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.
        y : ndarray, shape (n_trials,)
            Predicted target vector relative to X.

        Returns
        -------
        accuracy : double
            Accuracy of predictions from X with respect y.
        """
        return self._pipe.score(X, y)

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            Class labels for samples in X.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """Return the probabilities associated with predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, 0] == True if the nth sample is assigned to 1st class;
            prob[n, 1] == True if the nth sample is assigned to 2nd class.
        """

        return self._pipe.predict_proba(X)

    def transform(self, X):
        """Transform the data into feature vectors.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the data.
            the dimension of the feature vector depends on
            `n_filter` and `dim_red`.
        """
        return self._pipe.transform(X)
