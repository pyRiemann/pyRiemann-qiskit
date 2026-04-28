"""Quantum-enhanced SVM classifier."""

import numpy as np
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from scipy.special import softmax
from sklearn.svm import SVC

from ...utils.hyper_params_factory import gen_zz_feature_map
from ...utils.quantum_provider import get_quantum_kernel
from .quantic_classifier_base import QuanticClassifierBase


class QuanticSVM(QuanticClassifierBase):
    """Quantum-enhanced SVM classifier

    This class implements a support-vector machine (SVM) classifier [1]_,
    called SVC, on a quantum machine [2]_.
    Note that if `quantum` parameter is set to `False`
    then a classical SVC will be performed instead.

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
    .. versionchanged:: 0.6.0
        Migrate to Qiskit 2.x: use ``BackendSamplerV2`` via base class.
        Moved to :mod:`pyriemann_qiskit.classification.wrappers.quantic_svm`.

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
           'Supervised learning with quantum-enhanced feature spaces',
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    .. [3] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html

    .. [4] G. Gentinetta, A. Thomsen, D. Sutter, and S. Woerner,
           'The complexity of quantum support vector machines', arXiv,
           arXiv:2203.00031, Feb. 2022.
           doi: 10.48550/arXiv.2203.00031

    .. [5] S. Shalev-Shwartz, Y. Singer, and A. Cotter,
           'Pegasos: Primal Estimated sub-GrAdient SOlver for SVM'

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
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if isinstance(self._classifier, QSVC):
            probs = softmax(self.predict_proba(X))
            labels = [np.argmax(prob) for prob in probs]
        else:
            labels = self._predict(X)
        self._log("Prediction finished.")
        return self._map_indices_to_classes(labels)
