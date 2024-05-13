"""
Contains the base class for all quantum classifiers
as well as several quantum classifiers than can run
in several modes quantum/classical and simulated/real
quantum computer.
"""
from datetime import datetime
from pyriemann_qiskit.utils.quantum_provider import get_quantum_kernel
from scipy.special import softmax
import logging
import numpy as np
from warnings import warn

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM
from pyriemann_qiskit.datasets import get_feature_dimension
from pyriemann_qiskit.utils import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
    set_global_optimizer,
)
from pyriemann_qiskit.utils.distance import distance_functions
from pyriemann_qiskit.utils.utils import is_qfunction
from qiskit.primitives import BackendSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import QSVC, VQC, PegasosQSVC
from qiskit_optimization.algorithms import (
    CobylaOptimizer,
    SlsqpOptimizer,
)
from qiskit_algorithms.optimizers import SLSQP
from sklearn.svm import SVC

from .utils.hyper_params_factory import gen_zz_feature_map, gen_two_local, get_spsa
from .utils import get_provider, get_device, get_simulator
from .utils.distance import qdistance_logeuclid_to_convex_hull
from joblib import Parallel, delayed
import random

logging.basicConfig(level=logging.WARNING)


class QuanticClassifierBase(BaseEstimator, ClassifierMixin):

    """Quantum classifier

    This class implements a scikit-learn wrapper around Qiskit library [1]_.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:

    * there is no noise on a simulated quantum computer
      (so results are better),
    * a real quantum computer is quicker than a quantum simulator,
    * tasks on a real quantum computer are assigned to a queue
      before being executed on a back-end (delayed execution).

    Parameters
    ----------
    quantum : bool, default=True
        - If true will run on local or remote quantum backend
          (depending on q_account_token value),
        - If false, will perform classical computing instead.
    q_account_token : string | None, default=None
        If `quantum` is True and `q_account_token` provided,
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
    verbose : bool, default=True
        If true, will output all intermediate results and logs.
    shots : int, default=1024
        Number of repetitions of each circuit, for sampling.
    gen_feature_map : Callable[int, QuantumCircuit | FeatureMap], \
                      default=Callable[int, ZZFeatureMap]
        Function generating a feature map to encode data into a quantum state.
    seed : int | None, default=None
        Random seed for the simulation.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added support for multi-class classification.
    .. versionchanged:: 0.2.0
        Add seed parameter
    .. versionchanged:: 0.3.0
        Switch from IBMProvider to QiskitRuntimeService.

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    QuanticSVM
    QuanticVQC
    QuanticMDM

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    """

    def __init__(
        self,
        quantum=True,
        q_account_token=None,
        verbose=True,
        shots=1024,
        gen_feature_map=gen_zz_feature_map(),
        seed=None,
    ):
        self.verbose = verbose
        self._log("Initializing Quantum Classifier")
        self.q_account_token = q_account_token
        self.quantum = quantum
        self.shots = shots
        self.seed = datetime.now().microsecond if seed is None else seed
        self.gen_feature_map = gen_feature_map
        # protected field for child classes
        self._training_input = {}

    def _init_quantum(self):
        if self.quantum:
            if self.q_account_token:
                self._log("Real quantum computation will be performed")
                if not self.q_account_token == "load_account":
                    QiskitRuntimeService.delete_account()
                    QiskitRuntimeService.save_account(
                        channel="ibm_quantum",
                        token=self.q_account_token,
                        overwrite=True,
                    )
                self._log("Getting provider...")
                self._provider = get_provider()
            else:
                self._log("Quantum simulation will be performed")
                self._backend = get_simulator()
        else:
            self._log("Classical computation will be performed")

    def _log(self, *values):
        if self.verbose:
            print("[QClass] ", *values)

    def _split_classes(self, X, y):
        n_classes = len(self.classes_)
        X_classes = []
        for idx in range(n_classes):
            X_classes.append(X[y == self.classes_[idx]])
        return X_classes

    def _map_classes_to_indices(self, y):
        y_copy = y.copy()
        n_classes = len(self.classes_)
        for idx in range(n_classes):
            y_copy[y == self.classes_[idx]] = idx
        return y_copy

    def _map_indices_to_classes(self, y):
        y_copy = np.array(y.copy())
        n_classes = len(self.classes_)
        for idx in range(n_classes):
            y_copy[np.array(y).transpose() == idx] = self.classes_[idx]
        return np.array(y_copy)

    def fit(self, X, y):
        """Uses a quantum backend and fits the training data.

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

        X_classes = self._split_classes(X, y)
        y = self._map_classes_to_indices(y)

        n_classes = len(self.classes_)
        for idx in range(n_classes):
            self._training_input[self.classes_[idx]] = X_classes[idx]

        n_features = get_feature_dimension(self._training_input)
        self._log("Feature dimension = ", n_features)
        if hasattr(self, "gen_feature_map") and self.gen_feature_map is not None:
            self._feature_map = self.gen_feature_map(n_features)
        if self.quantum:
            if not hasattr(self, "_backend"):
                self._backend = get_device(self._provider, n_features)
            self._log("Quantum backend = ", self._backend)
            self._log("seed = ", self.seed)
            self._quantum_instance = BackendSampler(
                self._backend,
                options={"shots": self.shots, "seed_simulator": self.seed},
            )
            self._quantum_instance.transpile_options["seed_transpiler"] = self.seed
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
        y : ndarray, shape (n_samples,)
            Predicted target vector relative to X.

        Returns
        -------
        accuracy : double
            Accuracy of predictions from X with respect y.
        """
        y = self._map_classes_to_indices(y)
        self._log("Testing...")
        return self._classifier.score(X, y)

    def _predict(self, X):
        self._log("Prediction: ", X.shape)
        result = self._classifier.predict(X)
        self._log("Prediction finished.")
        return result

    def predict_proba(self, X):
        """Return the probabilities associated with predictions.

        The default behavior is to return the nested classifier probabilities.
        In case where no `predict_proba` method is available inside the classifier,
        the method predicts the label number (0 or 1 for examples) and applies a
        softmax in top of it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, i] == 1 if the nth sample is assigned to class `i`.
        """

        if not hasattr(self._classifier, "predict_proba"):
            # Classifier has no predict_proba
            # Use the result from predict and apply a softmax
            self._log(
                "No predict_proba method available.\
                       Computing softmax probabilities..."
            )
            proba = self._classifier.predict(X)
            proba = [
                np.array(
                    [
                        1 if c == self.classes_[i] else 0
                        for i in range(len(self.classes_))
                    ]
                )
                for c in proba
            ]
            proba = softmax(proba, axis=0)
        else:
            proba = self._classifier.predict_proba(X)

        return np.array(proba)


class QuanticSVM(QuanticClassifierBase):

    """Quantum-enhanced SVM classifier

    This class implements a support-vector machine (SVM) classifier [1]_,
    called SVC, on a quantum machine [2]_.
    Note that if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.0.2
        Qiskit's Pegasos implementation [4]_, [5]_.
    .. versionchanged:: 0.1.0
        Fix: copy estimator not keeping base class parameters.
    .. versionchanged:: 0.2.0
        Add seed parameter
        SVC and QSVC now compute probability (may impact performance)
        Predict is now using predict_proba with a softmax, when using QSVC.
    .. versionchanged:: 0.3.0
        Add use_fidelity_state_vector_kernel parameter

    Parameters
    ----------
    gamma : float | None, default=None
        Used as input for sklearn rbf_kernel which is used internally.
        See [3]_ for more information about gamma.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        Note, if pegasos is enabled you may want to consider
        larger values of C.
    max_iter: int | None, default=None
        Number of steps in Pegasos or (Q)SVC.
        If None, respective default values for Pegasos and SVC
        are used. The default value for Pegasos is 1000.
        For (Q)SVC it is -1 (that is not limit).
    pegasos : boolean, default=False
        If true, uses Qiskit's PegasosQSVC instead of QSVC.
    quantum : bool, default=True
        - If true will run on local or remote backend
          (depending on q_account_token value),
        - If false, will perform classical computing instead.
    q_account_token : string | None, default=None
        If `quantum` is True and `q_account_token` provided,
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
    verbose : bool, default=True
        If true, will output all intermediate results and logs.
    shots : int, default=1024
        Number of repetitions of each circuit, for sampling.
    gen_feature_map : Callable[int, QuantumCircuit | FeatureMap], \
                      default=Callable[int, ZZFeatureMap]
        Function generating a feature map to encode data into a quantum state.
    seed : int | None, default=None
        Random seed for the simulation
    use_fidelity_state_vector_kernel: boolean (default=True)
        if True, use a FidelitystatevectorKernel for simulation.

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

    def __init__(
        self,
        gamma="scale",
        C=1.0,
        max_iter=None,
        pegasos=False,
        quantum=True,
        q_account_token=None,
        verbose=True,
        shots=1024,
        gen_feature_map=gen_zz_feature_map(),
        seed=None,
        use_fidelity_state_vector_kernel=True,
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, gen_feature_map, seed
        )
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.pegasos = pegasos
        self.use_fidelity_state_vector_kernel = use_fidelity_state_vector_kernel

    def _init_algo(self, n_features):
        self._log("SVM initiating algorithm")
        if self.quantum:
            quantum_kernel = get_quantum_kernel(
                self._feature_map,
                self._quantum_instance,
                self.use_fidelity_state_vector_kernel,
            )
            if self.pegasos:
                self._log("[Warning] `gamma` is not supported by PegasosQSVC")
                num_steps = 1000 if self.max_iter is None else self.max_iter
                classifier = PegasosQSVC(
                    quantum_kernel=quantum_kernel, C=self.C, num_steps=num_steps
                )
            else:
                max_iter = -1 if self.max_iter is None else self.max_iter
                classifier = QSVC(
                    quantum_kernel=quantum_kernel,
                    gamma=self.gamma,
                    C=self.C,
                    max_iter=max_iter,
                    probability=True,
                )
        else:
            max_iter = -1 if self.max_iter is None else self.max_iter
            classifier = SVC(
                gamma=self.gamma, C=self.C, max_iter=max_iter, probability=True
            )
        return classifier

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
        if isinstance(self._classifier, QSVC):
            probs = softmax(self.predict_proba(X))
            labels = [np.argmax(prob) for prob in probs]
        else:
            labels = self._predict(X)
        self._log("Prediction finished.")
        return self._map_indices_to_classes(labels)


class QuanticVQC(QuanticClassifierBase):

    """Variational quantum classifier

    This class implements a variational quantum classifier (VQC).
    Note that there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not).

    Parameters
    ----------
    optimizer : Optimizer, default=SPSA
        The classical optimizer to use. See [3]_ for details.
    gen_var_form : Callable[int, QuantumCircuit | VariationalForm], \
                   default=Callable[int, TwoLocal]
        Function generating a variational form instance.
    quantum : bool, default=True
        - If true will run on local or remote backend
          (depending on q_account_token value).
        - If false, will perform classical computing instead.
    q_account_token : string | None, default=None
        If `quantum` is True and `q_account_token` provided,
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
    verbose : bool, default=True
        If true, will output all intermediate results and logs
    shots : int, default=1024
        Number of repetitions of each circuit, for sampling
    gen_feature_map : Callable[int, QuantumCircuit | FeatureMap], \
                      default=Callable[int, ZZFeatureMap]
        Function generating a feature map to encode data into a quantum state.
    seed : int | None, default=None
        Random seed for the simulation.

    Attributes
    ----------
    evaluated_values_ : list[int]
        Training curve values.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Fix: copy estimator not keeping base class parameters.
        Added support for multi-class classification.
    .. versionchanged:: 0.2.0
        Add seed parameter
    .. versionchanged:: 0.3.0
        Add evaluated_values_ attribute.

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

    def __init__(
        self,
        optimizer=get_spsa(),
        gen_var_form=gen_two_local(),
        quantum=True,
        q_account_token=None,
        verbose=True,
        shots=1024,
        gen_feature_map=gen_zz_feature_map(),
        seed=None,
    ):
        if quantum is False:
            raise ValueError(
                "VQC can only run on a quantum \
                              computer or simulator."
            )
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, gen_feature_map, seed
        )
        self.optimizer = optimizer
        self.gen_var_form = gen_var_form

    def _init_algo(self, n_features):
        self._log("VQC training...")
        var_form = self.gen_var_form(n_features)
        self.evaluated_values_ = []

        def _callback(_weights, value):
            self.evaluated_values_.append(value)

        vqc = VQC(
            optimizer=self.optimizer,
            feature_map=self._feature_map,
            ansatz=var_form,
            sampler=self._quantum_instance,
            num_qubits=n_features,
            callback=_callback,
        )
        return vqc

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
        return self._map_indices_to_classes(labels)

    @property
    def parameter_count(self):
        """Returns the number of parameters inside the variational circuit.
        This is determined by the `gen_var_form` attribute of this instance.

        Returns
        -------
        n_params : int
            The number of parameters in the variational circuit.
            Returns 0 if the instance is not fit yet.
        """

        if hasattr(self, "_classifier"):
            return len(self._classifier.ansatz.parameters)

        self._log("Instance not initialized. Parameter count is 0.")
        return 0


class QuanticMDM(QuanticClassifierBase):

    """Quantum-enhanced MDM classifier

    This class is a quantic implementation of the Minimum Distance to Mean
    (MDM) [1]_, which can run with quantum optimization.
    Only log-Euclidean distance between trial and class prototypes is supported
    at the moment, but any type of metric can be used for centroid estimation.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.1.0
        Fix: copy estimator not keeping base class parameters.
    .. versionchanged:: 0.2.0
        Add seed parameter.
        Add regularization parameter.
        Add classical_optimizer parameter.
    .. versionchanged:: 0.3.0
        Add qaoa_optimizer parameter.

    Parameters
    ----------
    metric : string | dict, default={"mean": 'logeuclid', \
            "distance": 'qlogeuclid_hull'}
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metrics for the centroid estimation and the
        distance estimation.
    quantum : bool, default=True
        Only applies if `metric` contains a quantic distance or mean.

        - If true will run on local or remote backend
          (depending on q_account_token value),
        - If false, will perform classical computing instead.
    q_account_token : string | None, default=None
        If `quantum` is True and `q_account_token` provided,
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
    verbose : bool, default=True
        If true, will output all intermediate results and logs.
    shots : int, default=1024
        Number of repetitions of each circuit, for sampling.
    seed : int | None, default=None
        Random seed for the simulation.
    upper_bound : int, default=7
        The maximum integer value for matrix normalization.
    regularization : MixinTransformer, default=None
        Additional post-processing to regularize means.
    classical_optimizer : OptimizationAlgorithm, default=CobylaOptimizer()
        An instance of OptimizationAlgorithm [3]_.
    qaoa_optimizer : SciPyOptimizer, default=SLSQP()
        An instance of a scipy optimizer to find the optimal weights for the
        parametric circuit (ansatz).

    See Also
    --------
    QuanticClassifierBase
    pyriemann.classification.MDM

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    .. [3] \
        https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.OptimizationAlgorithm.html#optimizationalgorithm
    """

    def __init__(
        self,
        metric={"mean": "logeuclid", "distance": "qlogeuclid_hull"},
        quantum=True,
        q_account_token=None,
        verbose=True,
        shots=1024,
        seed=None,
        upper_bound=7,
        regularization=None,
        classical_optimizer=CobylaOptimizer(rhobeg=2.1, rhoend=0.000001),
        qaoa_optimizer=SLSQP(),
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, None, seed
        )
        self.metric = metric
        self.upper_bound = upper_bound
        self.regularization = regularization
        self.classical_optimizer = classical_optimizer
        self.qaoa_optimizer = qaoa_optimizer

    @staticmethod
    def _override_predict_distance(mdm):
        """Override _predict_distances method of MDM.

        We override the _predict_distances method inside MDM to allow the use
        of qdistance.
        This is due to the fact the the signature of qdistances is different
        from the usual distance functions.
        """

        def _predict_distances(X):
            if is_qfunction(mdm.metric_dist):
                if "hull" in mdm.metric_dist:
                    warn("qdistances to hull should not be use inside MDM")
                else:
                    warn(
                        "q-distances for MDM are toy functions.\
                            Use pyRiemann distances instead."
                    )
                distance = distance_functions[mdm.metric_dist]
                centroids = np.array(mdm.covmeans_)
                weights = [distance(centroids, x) for x in X]
                return 1 - np.array(weights)
            else:
                return MDM._predict_distances(mdm, X)

        return _predict_distances

    def _init_algo(self, n_features):
        self._log("Quantic MDM initiating algorithm")
        classifier = MDM(metric=self.metric)
        classifier._predict_distances = QuanticMDM._override_predict_distance(
            classifier
        )
        if self.quantum:
            self._log("Using NaiveQAOAOptimizer")
            self._optimizer = NaiveQAOAOptimizer(
                quantum_instance=self._quantum_instance,
                upper_bound=self.upper_bound,
                optimizer=self.qaoa_optimizer,
            )
        else:
            self._log("Using ClassicalOptimizer (COBYLA)")
            self._optimizer = ClassicalOptimizer(self.classical_optimizer)
        set_global_optimizer(self._optimizer)
        return classifier

    def _train(self, X, y):
        QuanticClassifierBase._train(self, X, y)
        if self.regularization is not None:
            self._classifier.covmeans_ = self.regularization.fit_transform(
                self._classifier.covmeans_
            )

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
        return self._map_indices_to_classes(labels)


class NearestConvexHull(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Nearest Convex Hull Classifier (NCH)

    In NCH, for each class a convex hull is produced by the set of matrices
    corresponding to each class. There is no training. Calculating a distance
    to a hull is an optimization problem and it is calculated for each testing
    sample (SPD matrix) and each hull/class. The minimal distance defines the
    predicted class.

    Notes
    -----
    .. versionadded:: 0.2.0

    Parameters
    ----------
    n_jobs : int, default=6
        The number of jobs to use for the computation. This works by computing
        each of the hulls in parallel.
    n_hulls_per_class : int, default=3
        The number of hulls used per class, when subsampling is "random".
    n_samples_per_hull : int, default=15
        Defines how many samples are used to build a hull. -1 will include
        all samples per class.
    subsampling : {"min", "random"}, default="min"
        Subsampling strategy of training set to estimate distance to hulls.
        "min" estimates hull using the n_samples_per_hull closest matrices.
        "random" estimates hull using n_samples_per_hull random matrices.

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        ‘Convex Class Model on Symmetric Positive Definite Manifolds’,
        Image and Vision Computing, 2019.
    """

    def __init__(
        self,
        n_jobs=6,
        n_hulls_per_class=3,
        n_samples_per_hull=10,
        subsampling="min",
    ):
        """Init."""
        self.n_jobs = n_jobs
        self.n_samples_per_hull = n_samples_per_hull
        self.n_hulls_per_class = n_hulls_per_class
        self.matrices_per_class_ = {}
        self.debug = False
        self.subsampling = subsampling

        if subsampling not in ["min", "random"]:
            raise ValueError(f"Unknown subsampling type {subsampling}.")

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : NearestConvexHull instance
            The NearestConvexHull instance.
        """

        if self.debug:
            print("Start NCH Train")
        self.classes_ = np.unique(y)

        for c in self.classes_:
            self.matrices_per_class_[c] = X[y == c]

        if self.debug:
            print("Samples per class:")
            for c in self.classes_:
                print("Class: ", c, " Count: ", self.matrices_per_class_[c].shape[0])

            print("End NCH Train")

    def _process_sample_min_hull(self, x):
        """Finds the closes N covmats and uses them to build a single hull per class"""
        dists = []

        for c in self.classes_:
            dist = distance(self.matrices_per_class_[c], x, metric="logeuclid")[:, 0]
            # take the closest matrices
            indexes = np.argsort(dist)[0 : self.n_samples_per_hull]

            if self.debug:
                print("Distances to test sample: ", dist)
                print("Smallest N distances indexes:", indexes)
                print("Smallest N distances: ")
                for pp in indexes:
                    print(dist[pp])

            d = qdistance_logeuclid_to_convex_hull(
                self.matrices_per_class_[c][indexes], x
            )

            if self.debug:
                print("Final hull distance:", d)

            dists.append(d)

        return dists

    def _process_sample_random_hull(self, x):
        """Uses random samples to build a hull, can be several hulls per class"""
        dists = []

        for c in self.classes_:
            dist_total = 0

            # using multiple hulls
            for i in range(0, self.n_hulls_per_class):
                if self.n_samples_per_hull == -1:  # use all data per class
                    hull_data = self.matrices_per_class_[c]
                else:  # use a subset of the data per class
                    random_samples = random.sample(
                        range(self.matrices_per_class_[c].shape[0]),
                        k=self.n_samples_per_hull,
                    )
                    hull_data = self.matrices_per_class_[c][random_samples]

                dist = qdistance_logeuclid_to_convex_hull(hull_data, x)
                dist_total = dist_total + dist

            dists.append(dist_total)

        return dists

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        dists = []

        if self.debug:
            print("Total test samples:", X.shape[0])

        if self.subsampling == "min":
            self._process_sample = self._process_sample_min_hull
        elif self.subsampling == "random":
            self._process_sample = self._process_sample_random_hull
        else:
            raise ValueError(f"Unknown subsampling type {self.subsampling}.")

        parallel = self.n_jobs > 1

        if self.debug:
            if parallel:
                print("Running in parallel")
            else:
                print("Not running in parallel")

        if parallel:
            dists = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_sample)(x) for x in X
            )

        else:
            for x in X:
                dist = self._process_sample(x)
                dists.append(dist)

        dists = np.asarray(dists)

        return dists

    def predict(self, X):
        """Get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest convex hull.
        """
        if self.debug:
            print("Start NCH Predict")
        dist = self._predict_distances(X)

        predictions = self.classes_[dist.argmin(axis=1)]

        if self.debug:
            print("End NCH Predict")

        return predictions

    def transform(self, X):
        """Get the distance to each convex hull.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The distance to each convex hull.
        """

        if self.debug:
            print("NCH Transform")
        return self._predict_distances(X)


class QuanticNCH(QuanticClassifierBase):

    """A Quantum wrapper around the NCH algorithm.

    It allows both classical and Quantum versions to be executed.

    Notes
    -----
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.3.0
        Add qaoa_optimizer parameter.

    Parameters
    ----------
    quantum : bool, default=True
        Only applies if `metric` contains a cpm distance or mean.

        - If true will run on local or remote backend
          (depending on q_account_token value),
        - If false, will perform classical computing instead.
    q_account_token : string | None, default=None
        If `quantum` is True and `q_account_token` provided,
        the classification task will be running on a IBM quantum backend.
        If `load_account` is provided, the classifier will use the previous
        token saved with `IBMProvider.save_account()`.
    verbose : bool, default=True
        If true, will output all intermediate results and logs.
    shots : int, default=1024
        Number of repetitions of each circuit, for sampling.
    seed : int | None, default=None
        Random seed for the simulation
    upper_bound : int, default=7
        The maximum integer value for matrix normalization.
    regularization : MixinTransformer | None, default=None
        Additional post-processing to regularize means.
    classical_optimizer : OptimizationAlgorithm, default=SlsqpOptimizer()
        An instance of OptimizationAlgorithm [3]_.
    n_jobs : int, default=6
        The number of jobs to use for the computation. This works by computing
        each of the hulls in parallel.
    n_hulls_per_class : int, default=3
        The number of hulls used per class.
    n_samples_per_hull : int, default=15
        Defines how many samples are used to build a hull. -1 will include
        all samples per class.
    subsampling : {"min", "random"}, default="min"
        Subsampling strategy of training set to estimate distance to hulls.
        "min" estimates hull using the n_samples_per_hull closest matrices.
        "random" estimates hull using n_samples_per_hull random matrices.
    qaoa_optimizer : SciPyOptimizer, default=SLSQP()
        An instance of a scipy optimizer to find the optimal weights for the
        parametric circuit (ansatz).

    """

    def __init__(
        self,
        quantum=True,
        q_account_token=None,
        verbose=True,
        shots=1024,
        seed=None,
        upper_bound=7,
        regularization=None,
        n_jobs=6,
        classical_optimizer=SlsqpOptimizer(),  # set here new default optimizer
        n_hulls_per_class=3,
        n_samples_per_hull=10,
        subsampling="min",
        qaoa_optimizer=SLSQP(),
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, None, seed
        )
        self.upper_bound = upper_bound
        self.regularization = regularization
        self.classical_optimizer = classical_optimizer
        self.n_hulls_per_class = n_hulls_per_class
        self.n_samples_per_hull = n_samples_per_hull
        self.n_jobs = n_jobs
        self.subsampling = subsampling
        self.qaoa_optimizer = qaoa_optimizer

    def _init_algo(self, n_features):
        self._log("Nearest Convex Hull Classifier initiating algorithm")

        classifier = NearestConvexHull(
            n_hulls_per_class=self.n_hulls_per_class,
            n_samples_per_hull=self.n_samples_per_hull,
            n_jobs=self.n_jobs,
            subsampling=self.subsampling,
        )

        if self.quantum:
            self._log("Using NaiveQAOAOptimizer")
            self._optimizer = NaiveQAOAOptimizer(
                quantum_instance=self._quantum_instance,
                upper_bound=self.upper_bound,
                optimizer=self.qaoa_optimizer,
            )
        else:
            self._log("Using ClassicalOptimizer")
            self._optimizer = ClassicalOptimizer(self.classical_optimizer)

        # sets the optimizer for the distance functions
        # used in NearestConvexHull class
        set_global_optimizer(self._optimizer)

        return classifier

    def predict(self, X):
        # self._log("QuanticNCH Predict")
        return self._predict(X)

    def transform(self, X):
        # self._log("QuanticNCH Transform")
        return self._classifier.transform(X)
