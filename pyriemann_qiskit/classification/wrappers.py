"""
Contains the base class for all quantum classifiers
as well as several quantum classifiers than can run
in several modes quantum/classical and simulated/real
quantum computer.
"""
import logging
from datetime import datetime

import numpy as np
from qiskit.primitives import BackendSampler
from qiskit_algorithms.optimizers import SLSQP
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import QSVC, VQC, PegasosQSVC
from qiskit_optimization.algorithms import CobylaOptimizer, SlsqpOptimizer
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from ..datasets import get_feature_dimension
from ..utils.docplex import set_global_optimizer
from ..utils.hyper_params_factory import gen_two_local, gen_zz_feature_map, get_spsa
from ..utils.quantum_provider import (
    get_device,
    get_provider,
    get_quantum_kernel,
    get_simulator,
)
from ..utils.utils import get_docplex_optimizer_from_params_bag
from .algorithms import CpMDM, NearestConvexHull

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
    gen_feature_map : Callable[[int, str], QuantumCircuit | FeatureMap], \
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
    .. versionchanged:: 0.4.0
        Add n_jobs and use_qiskit_symb parameter
        for SymbFidelityStatevectorKernel

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
    gen_feature_map : Callable[[int, str], QuantumCircuit | FeatureMap], \
                      default=Callable[int, ZZFeatureMap]
        Function generating a feature map to encode data into a quantum state.
    seed : int | None, default=None
        Random seed for the simulation
    use_fidelity_state_vector_kernel: boolean, default=True
        if True, use a FidelitystatevectorKernel for simulation.
    use_qiskit_symb: boolean, default=True
        This flag is used only if qiskit-symb is installed, and pegasos is False.
        If True and the number of qubits < 9, then qiskit_symb is used.
    n_jobs: boolean
        The number of jobs for the qiskit-symb fidelity state vector
        (if applicable)

    See Also
    --------
    QuanticClassifierBase
    SymbFidelityStatevectorKernel

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
        use_qiskit_symb=True,
        n_jobs=4,
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, gen_feature_map, seed
        )
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.pegasos = pegasos
        self.use_fidelity_state_vector_kernel = use_fidelity_state_vector_kernel
        self.n_jobs = n_jobs
        self.use_qiskit_symb = use_qiskit_symb

    def _init_algo(self, n_features):
        self._log("SVM initiating algorithm")
        if self.quantum:
            quantum_kernel = get_quantum_kernel(
                self._feature_map,
                self.gen_feature_map,
                self._quantum_instance,
                self.use_fidelity_state_vector_kernel,
                self.use_qiskit_symb and not self.pegasos,
                self.n_jobs,
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
    gen_feature_map : Callable[[int, str], QuantumCircuit | FeatureMap], \
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
        Add `evaluated_values_` attribute.

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
    .. versionchanged:: 0.4.0
        Add QAOA-CV optimization.
    .. versionchanged:: 0.4.1
        Add qaoa_initial_points parameter.
    .. versionchanged:: 0.4.2
        Separate wrapper from algorithm (CpMDM)

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
    create_mixer : None | Callable[int, QuantumCircuit], default=None
        A delegate that takes into input an angle and returns a QuantumCircuit.
        This circuit is the mixer operatior for the QAOA-CV algorithm.
        If None and quantum, the NaiveQAOAOptimizer will be used instead.
    n_reps : int, default=3
        The number of time the mixer and cost operator are repeated in the QAOA-CV
        circuit.
    qaoa_initial_points : Tuple[int, int], default=[0.0, 0.0].
        Starting parameters (beta and gamma) for the NaiveQAOAOptimizer.

    See Also
    --------
    QuanticClassifierBase
    classification.algorithms.CpMDM
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
        create_mixer=None,
        n_reps=3,
        qaoa_initial_points=[0.0, 0.0],
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, None, seed
        )
        self.metric = metric
        self.upper_bound = upper_bound
        self.regularization = regularization
        self.classical_optimizer = classical_optimizer
        self.qaoa_optimizer = qaoa_optimizer
        self.create_mixer = create_mixer
        self.n_reps = n_reps
        self.qaoa_initial_points = qaoa_initial_points

    def _init_algo(self, n_features):
        self._log("Quantic MDM initiating algorithm")
        classifier = CpMDM(metric=self.metric)
        self._optimizer = get_docplex_optimizer_from_params_bag(
            self,
            self.quantum,
            self._quantum_instance if self.quantum else None,
            self.upper_bound,
            self.qaoa_optimizer,
            self.classical_optimizer,
            self.create_mixer,
            self.n_reps,
            self.qaoa_initial_points,
        )
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


class QuanticNCH(QuanticClassifierBase):

    """A Quantum wrapper around the NCH algorithm.

    It allows both classical and Quantum versions to be executed.

    Notes
    -----
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.3.0
        Add qaoa_optimizer parameter.
    .. versionchanged:: 0.4.0
        Add QAOA-CV optimization.
    .. versionchanged:: 0.4.1
        Add the qaoa_initial_points parameter.

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
        An instance of OptimizationAlgorithm [1]_.
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
    create_mixer : None | Callable[int, QuantumCircuit], default=None
        A delegate that takes into input an angle and returns a QuantumCircuit.
        This circuit is the mixer operatior for the QAOA-CV algorithm.
        If None and quantum, the NaiveQAOAOptimizer will be used instead.
    n_reps : int, default=3
        The number of time the mixer and cost operator are repeated in the QAOA-CV
        circuit.
    qaoa_initial_points : Tuple[int, int], default=[0.0, 0.0].
        Starting parameters (beta and gamma) for the NaiveQAOAOptimizer.

    References
    ----------
    .. [1] \
        https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.OptimizationAlgorithm.html#optimizationalgorithm
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
        create_mixer=None,
        n_reps=3,
        qaoa_initial_points=[0.0, 0.0],
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
        self.create_mixer = create_mixer
        self.n_reps = n_reps
        self.qaoa_initial_points = qaoa_initial_points

    def _init_algo(self, n_features):
        self._log("Nearest Convex Hull Classifier initiating algorithm")

        classifier = NearestConvexHull(
            n_hulls_per_class=self.n_hulls_per_class,
            n_samples_per_hull=self.n_samples_per_hull,
            n_jobs=self.n_jobs,
            subsampling=self.subsampling,
            seed=self.seed,
        )

        self._optimizer = get_docplex_optimizer_from_params_bag(
            self,
            self.quantum,
            self._quantum_instance if self.quantum else None,
            self.upper_bound,
            self.qaoa_optimizer,
            self.classical_optimizer,
            self.create_mixer,
            self.n_reps,
            self.qaoa_initial_points,
        )

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
