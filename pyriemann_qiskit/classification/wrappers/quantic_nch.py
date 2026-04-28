"""Quantum wrapper around the NCH algorithm."""

from qiskit_algorithms.optimizers import SLSQP
from qiskit_optimization.algorithms import SlsqpOptimizer

from ...utils.utils import get_docplex_optimizer_from_params_bag
from ..algorithms import NearestConvexHull
from .quantic_classifier_base import QuanticClassifierBase


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
    .. versionchanged:: 0.5.0
        Add the qaoacv_implementation parameter.
    .. versionchanged:: 0.6.0
        Moved to :mod:`pyriemann_qiskit.classification.wrappers.quantic_nch`.

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
        select QAOACVOptimizer. If None, uses default QAOA-CV behavior.

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
        classical_optimizer=None,
        n_hulls_per_class=3,
        n_samples_per_hull=10,
        subsampling="min",
        qaoa_optimizer=None,
        create_mixer=None,
        n_reps=3,
        qaoa_initial_points=None,
        qaoacv_implementation=None,
    ):
        QuanticClassifierBase.__init__(
            self, quantum, q_account_token, verbose, shots, None, seed
        )
        self.upper_bound = upper_bound
        self.regularization = regularization
        self.classical_optimizer = (
            classical_optimizer if classical_optimizer is not None else SlsqpOptimizer()
        )
        self.n_hulls_per_class = n_hulls_per_class
        self.n_samples_per_hull = n_samples_per_hull
        self.n_jobs = n_jobs
        self.subsampling = subsampling
        self.qaoa_optimizer = qaoa_optimizer if qaoa_optimizer is not None else SLSQP()
        self.create_mixer = create_mixer
        self.n_reps = n_reps
        self.qaoa_initial_points = (
            qaoa_initial_points if qaoa_initial_points is not None else [0.0, 0.0]
        )
        self.qaoacv_implementation = qaoacv_implementation

    def _init_algo(self, n_features):
        self._log("Nearest Convex Hull Classifier initiating algorithm")

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

        classifier = NearestConvexHull(
            n_hulls_per_class=self.n_hulls_per_class,
            n_samples_per_hull=self.n_samples_per_hull,
            n_jobs=self.n_jobs,
            subsampling=self.subsampling,
            seed=self.seed,
            optimizer=self._optimizer,
        )

        return classifier

    def predict(self, X):
        # self._log("QuanticNCH Predict")
        return self._predict(X)

    def transform(self, X):
        # self._log("QuanticNCH Transform")
        return self._classifier.transform(X)
