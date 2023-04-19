"""Module for classification function."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from qiskit import BasicAer
from qiskit_ibm_provider import IBMProvider, least_busy
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.quantum_instance import logger
from qiskit_machine_learning.algorithms import QSVC, VQC, PegasosQSVC
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from datetime import datetime
import logging
from .utils.hyper_params_factory import (gen_zz_feature_map,
                                         gen_two_local,
                                         get_spsa)
from .utils import get_provider, get_devices
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.datasets import get_feature_dimension

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
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
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
            algorithm_globals.random_seed = datetime.now().microsecond
            self._log("seed = ", algorithm_globals.random_seed)
            if self.q_account_token:
                self._log("Real quantum computation will be performed")
                if not self.q_account_token == "load_account":
                    IBMProvider.delete_account()
                    IBMProvider.save_account(self.q_account_token)
                IBMProvider.load_account()
                self._log("Getting provider...")
                self._provider = get_provider()
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

        class1, class0 = self._split_classes(X, y)
        y = self._map_classes_to_0_1(y)

        self._training_input[self.classes_[1]] = class1
        self._training_input[self.classes_[0]] = class0

        n_features = get_feature_dimension(self._training_input)
        self._log("Feature dimension = ", n_features)
        self._feature_map = self.gen_feature_map(n_features)
        if self.quantum:
            if not hasattr(self, "_backend"):
                devices = get_devices(self._provider, n_features)
                try:
                    self._backend = least_busy(devices)
                except Exception:
                    self._log("Devices are all busy. Getting the first one...")
                    self._backend = devices[0]
                self._log("Quantum backend = ", self._backend)
            seed_sim = algorithm_globals.random_seed
            seed_trs = algorithm_globals.random_seed
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
        self._classifier.fit(X, y)

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
        return self._classifier.score(X, y)

    def _predict(self, X):
        self._log("Prediction: ", X.shape)
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
    .. versionchanged:: 0.0.2
       Qiskit's Pegasos implementation [4, 5]_.

    Parameters
    ----------
    gamma : float | None (default: None)
        Used as input for sklearn rbf_kernel which is used internally.
        See [3]_ for more information about gamma.
    C : float (default: 1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        Note, if pegasos is enabled you may want to consider
        larger values of C.
    max_iter: int | None (default: None)
        number of steps in Pegasos or (Q)SVC.
        If None, respective default values for Pegasos and SVC
        are used. The default value for Pegasos is 1000.
        For (Q)SVC it is -1 (that is not limit).
    pegasos : boolean (default: False)
        If true, uses Qiskit's PegasosQSVC instead of QSVC.

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

    .. [4] G. Gentinetta, A. Thomsen, D. Sutter, and S. Woerner,
           ‘The complexity of quantum support vector machines’, arXiv,
           arXiv:2203.00031, Feb. 2022.
           doi: 10.48550/arXiv.2203.00031

    .. [5] S. Shalev-Shwartz, Y. Singer, and A. Cotter,
           ‘Pegasos: Primal Estimated sub-GrAdient SOlver for SVM’

    """

    def __init__(self, gamma='scale', C=1.0, max_iter=None,
                 pegasos=False, **parameters):
        QuanticClassifierBase.__init__(self, **parameters)
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.pegasos = pegasos

    def _init_algo(self, n_features):
        self._log("SVM initiating algorithm")
        if self.quantum:
            quantum_kernel = \
                QuantumKernel(feature_map=self._feature_map,
                              quantum_instance=self._quantum_instance)
            if self.pegasos:
                self._log("[Warning] `gamma` is not supported by PegasosQSVC")
                num_steps = 1000 if self.max_iter is None else self.max_iter
                classifier = PegasosQSVC(quantum_kernel=quantum_kernel,
                                         C=self.C,
                                         num_steps=num_steps)
            else:
                max_iter = -1 if self.max_iter is None else self.max_iter
                classifier = QSVC(quantum_kernel=quantum_kernel,
                                  gamma=self.gamma, C=self.C,
                                  max_iter=max_iter)
        else:
            max_iter = -1 if self.max_iter is None else self.max_iter
            classifier = SVC(gamma=self.gamma, C=self.C, max_iter=max_iter)
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
        https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html

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
        vqc = VQC(optimizer=self.optimizer,
                  feature_map=self._feature_map,
                  ansatz=var_form,
                  quantum_instance=self._quantum_instance,
                  num_qubits=n_features)
        return vqc

    def _map_classes_to_0_1(self, y):
        # Label must be one-hot encoded for VQC
        y_copy = np.ndarray((y.shape[0], 2))
        y_copy[y == self.classes_[0]] = [1, 0]
        y_copy[y == self.classes_[1]] = [0, 1]
        return y_copy

    def _map_0_1_to_classes(self, y):
        # Decode one-hot encoded labels
        y_copy = np.ndarray((y.shape[0], 1))
        y_copy[(y == [1, 0]).all()] = self.classes_[0]
        y_copy[(y == [0, 1]).all()] = self.classes_[1]
        return y_copy

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
        labels = self._predict(X)
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
    C : float (default: 1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        Note, if pegasos is enabled you may want to consider
        larger values of C.
    max_iter: int | None (default: None)
        number of steps in Pegasos or SVC.
        If None, respective default values for Pegasos and (Q)SVC
        are used. The default value for Pegasos is 1000.
        For (Q)SVC it is -1 (that is not limit).
    shots : int | None (default: 1024)
        Number of repetitions of each circuit, for sampling.
        If None, classical computation will be performed.
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
    params: Dict (default: {})
        Additional parameters to pass to the nested instance
        of the quantum classifier.
        See QuanticClassifierBase, QuanticVQC and QuanticSVM for
        a complete list of the parameters.

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
    QuanticClassifierBase

    References
    ----------
    .. [1] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html

    .. [2] \
        https://qiskit.org/documentation/stable/0.36/stubs/qiskit.circuit.library.NLocal.html

    """

    def __init__(self, nfilter=1, dim_red=PCA(),
                 gamma='scale', C=1.0, max_iter=None,
                 shots=1024, feature_entanglement='full',
                 feature_reps=2, spsa_trials=None, two_local_reps=None,
                 params={}):

        self.nfilter = nfilter
        self.dim_red = dim_red
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.shots = shots
        self.feature_entanglement = feature_entanglement
        self.feature_reps = feature_reps
        self.spsa_trials = spsa_trials
        self.two_local_reps = two_local_reps
        self.params = params

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
            clf = QuanticSVM(quantum=is_quantum, gamma=gamma, C=C,
                             max_iter=max_iter,
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
