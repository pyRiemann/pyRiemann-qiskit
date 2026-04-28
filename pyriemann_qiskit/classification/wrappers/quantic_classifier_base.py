"""Base class for quantum classifiers."""

import logging
from datetime import datetime

import numpy as np
from qiskit.primitives import BackendSamplerV2
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin

from ...datasets import get_feature_dimension
from ...utils.hyper_params_factory import gen_zz_feature_map
from ...utils.quantum_provider import get_device, get_provider, get_simulator

logging.basicConfig(level=logging.WARNING)


class QuanticClassifierBase(ClassifierMixin, BaseEstimator):
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
    .. versionchanged:: 0.6.0
        Migrate to Qiskit 2.x: replace ``BackendSampler`` with ``BackendSamplerV2``.
        Moved to :mod:`pyriemann_qiskit.classification.wrappers.quantic_classifier_base`.

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
                    from qiskit_ibm_runtime import QiskitRuntimeService

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
        """Fit the quantum classifier using training data.

        Initializes the quantum backend (simulator or real hardware) and trains
        the classifier on the provided data. The feature map is generated based
        on the number of features in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples,)
            Target labels corresponding to X.

        Returns
        -------
        self : QuanticClassifierBase instance
            The fitted classifier instance.
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
            self._quantum_instance = BackendSamplerV2(
                backend=self._backend,
                options={"default_shots": self.shots, "seed_simulator": self.seed},
            )
        self._classifier = self._init_algo(n_features)
        self._train(X, y)
        return self

    def _init_algo(self, n_features):
        raise Exception("Init algo method was not implemented")

    def _train(self, X, y):
        self._log("Training...")
        self._classifier.fit(X, y)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.
        y : ndarray, shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y.

        Notes
        -----
        For alternative metrics, use sklearn.model_selection.cross_val_score
        with a custom scoring parameter.
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
        """Predict class probabilities for X.

        Returns the probability of each class for each sample. If the underlying
        classifier does not implement predict_proba, predictions are converted
        to one-hot encoding and softmax is applied.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            Class probabilities for each sample. prob[i, j] is the probability
            that sample i belongs to class j.
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
