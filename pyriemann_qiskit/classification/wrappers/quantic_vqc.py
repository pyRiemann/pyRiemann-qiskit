"""Variational quantum classifier."""

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.algorithms import VQC

from ...utils.hyper_params_factory import gen_two_local, gen_zz_feature_map, get_spsa
from .quantic_classifier_base import QuanticClassifierBase


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
    .. versionchanged:: 0.6.0
        Pass ``pass_manager`` to ``VQC`` for Qiskit 2.x transpilation.
        Moved to :mod:`pyriemann_qiskit.classification.wrappers.quantic_vqc`.

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
           'Supervised learning with quantum-enhanced feature spaces',
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

        def _callback(*args):
            # qiskit-algorithms >= 0.4.0 with SPSA: (nfev, x, fx, dx, accept)
            # earlier / other optimizers: (weights, value)
            value = args[2] if len(args) == 5 else args[-1]
            self.evaluated_values_.append(value)

        pass_manager = None
        if hasattr(self, "_quantum_instance") and self._quantum_instance is not None:
            pass_manager = generate_preset_pass_manager(
                optimization_level=1, backend=self._quantum_instance.backend
            )
        vqc = VQC(
            optimizer=self.optimizer,
            feature_map=self._feature_map,
            ansatz=var_form,
            sampler=self._quantum_instance,
            num_qubits=n_features,
            callback=_callback,
            pass_manager=pass_manager,
        )
        return vqc

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
