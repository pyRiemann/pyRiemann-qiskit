"""Quantum-enhanced MDM classifier."""

from qiskit_algorithms.optimizers import SLSQP
from qiskit_optimization.algorithms import CobylaOptimizer

from ...utils.utils import get_docplex_optimizer_from_params_bag
from ..algorithms import CpMDM
from .quantic_classifier_base import QuanticClassifierBase


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
    .. versionchanged:: 0.5.0
        Add the qaoacv_implementation parameter.
    .. versionchanged:: 0.6.0
        Moved to :mod:`pyriemann_qiskit.classification.wrappers.quantic_mdm`.

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
        This circuit is the mixer operator for the QAOA-CV algorithm.
        If None and quantum, the NaiveQAOAOptimizer will be used instead.
    n_reps : int, default=3
        The number of time the mixer and cost operator are repeated in the QAOA-CV
        circuit.
    qaoa_initial_points : Tuple[int, int], default=[0.0, 0.0].
        Starting parameters (beta and gamma) for the NaiveQAOAOptimizer.
    qaoacv_implementation : {"ulvi", "luna"} | None, default=None
        QAOA-CV implementation variant. When create_mixer is provided:
        "ulvi" selects QAOACVAngleOptimizer, "luna" or other string values
        select QAOACVOptimizer. If None, uses default QAOACVOptimizer.
        If not quantum or not mixer provided, has no effect.

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
        classical_optimizer=None,
        qaoa_optimizer=None,
        create_mixer=None,
        n_reps=3,
        qaoa_initial_points=None,
        qaoacv_implementation=None,
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, None, seed
        )
        self.metric = metric
        self.upper_bound = upper_bound
        self.regularization = regularization
        self.classical_optimizer = (
            classical_optimizer
            if classical_optimizer is not None
            else CobylaOptimizer(rhobeg=2.1, rhoend=0.000001)
        )
        self.qaoa_optimizer = qaoa_optimizer if qaoa_optimizer is not None else SLSQP()
        self.create_mixer = create_mixer
        self.n_reps = n_reps
        self.qaoa_initial_points = (
            qaoa_initial_points if qaoa_initial_points is not None else [0.0, 0.0]
        )
        self.qaoacv_implementation = qaoacv_implementation

    def _init_algo(self, n_features):
        self._log("Quantic MDM initiating algorithm")

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
            self.qaoacv_implementation,
        )

        classifier = CpMDM(optimizer=self._optimizer, metric=self.metric)

        return classifier

    def _train(self, X, y):
        QuanticClassifierBase._train(self, X, y)
        if self.regularization is not None:
            self._classifier.covmeans_ = self.regularization.fit_transform(
                self._classifier.covmeans_
            )

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
        labels = self._predict(X)
        return self._map_indices_to_classes(labels)
