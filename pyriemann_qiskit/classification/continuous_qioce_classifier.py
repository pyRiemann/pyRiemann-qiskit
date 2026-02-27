"""QAOA Batch Classifier with Angle Encoding.

This module implements a quantum classifier that trains a single QAOA circuit
on all training vectors simultaneously, rather than optimizing each component
independently.
"""

import time

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import L_BFGS_B
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_random_state

from ..utils.docplex import QAOACVAngleOptimizer, create_mixer_rotational_X_gates


class ContinuousQIOCEClassifier(QAOACVAngleOptimizer, ClassifierMixin):
    """QAOA classifier with batch training using angle encoding.

    This classifier inherits from QAOACVAngleOptimizer and trains a single
    QAOA circuit on all training vectors simultaneously. Each component of
    the input vector is encoded as a qubit, and the circuit learns to map
    input patterns to class labels.

    Unlike QAOACVAngleOptimizer which optimizes each component independently,
    this classifier trains on the entire training set at once, learning
    discriminative patterns for classification.

    Parameters
    ----------
    n_reps : int, default=3
        Number of QAOA repetitions (layers).
    optimizer : Optimizer, default=L_BFGS_B()
        Classical optimizer for circuit parameters. L-BFGS-B is recommended
        for its efficiency with gradient-based optimization.
    create_mixer : callable, default=create_mixer_rotational_X_gates(0)
        Function to create mixer operator.
    max_features : int, default=10
        Maximum number of features (qubits) to use.
        If input has more features, dimensionality reduction should be applied.
    quantum_instance : QuantumInstance, default=None
        Quantum backend instance. If None, uses statevector simulation.
    random_state : int, RandomState or None, default=None
        Seed for reproducible parameter initialisation (sklearn convention).

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_ : int
        Number of features in training data.
    optim_params_ : ndarray
        Optimised γ (cost/mixer) circuit parameters.
    training_loss_history_ : list
        Loss values during training.
    X_min_ : ndarray
        Minimum values for feature normalisation.
    X_max_ : ndarray
        Maximum values for feature normalisation.
    X_train_ : ndarray
        Normalised training data.
    y_train_ : ndarray
        Training labels (binary: 0 or 1).
    state_vector_ : Statevector
        State vector at optimal parameters for the first training sample.
        Stored for interface compatibility with the parent class; not used
        in prediction.

    Notes
    -----
    .. versionadded:: 0.5.0

    Examples
    --------
    >>> from pyriemann_qiskit.classification import ContinuousQIOCEClassifier
    >>> clf = ContinuousQIOCEClassifier(n_reps=2, max_features=5)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_reps=3,
        optimizer=None,
        create_mixer=None,
        max_features=10,
        quantum_instance=None,
        random_state=None,
    ):
        # Initialize parent QAOACVAngleOptimizer
        super().__init__(
            create_mixer=(
                create_mixer if create_mixer is not None
                else create_mixer_rotational_X_gates(0)
            ),
            n_reps=n_reps,
            quantum_instance=quantum_instance,
            optimizer=optimizer if optimizer is not None else L_BFGS_B(),
        )
        self.max_features = max_features
        self.random_state = random_state

    def _normalize_features(self, X):
        """Normalize features to [0, pi] range for angle encoding."""
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)

        # Avoid division by zero
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0

        # Normalize to [0, 1] then scale to [0, pi]
        X_norm = (X - X_min) / X_range
        return X_norm * np.pi

    def _fit_normalize(self, X):
        """Store normalisation statistics and return normalised X.

        Computes ``X_min_`` and ``X_max_`` once and normalises in the same
        pass, so ``fit`` does not need to compute min/max twice.
        """
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)
        X_range = self.X_max_ - self.X_min_
        X_range[X_range == 0] = 1.0
        return (X - self.X_min_) / X_range * np.pi

    def _extract_class_probability(self, state_vec, n_features):
        """Extract class probability from quantum state.

        Computes the average Bloch Z-component directly from statevector
        amplitudes — O(2^n) instead of the O(n × 4^n) ``partial_trace``
        approach used in the parent class.

        Parameters
        ----------
        state_vec : Statevector
            Quantum state to evaluate.
        n_features : int
            Number of qubits.

        Returns
        -------
        float
            Predicted probability for class 1, in [0, 1].
        """
        amps = state_vec.data                # shape (2**n_features,)
        probs = np.abs(amps) ** 2            # probability per basis state
        indices = np.arange(len(probs))
        bloch_z_values = np.array([
            2.0 * probs[(indices >> i) & 1 == 0].sum() - 1.0
            for i in range(n_features)
        ])
        avg_bloch_z = bloch_z_values.mean()
        return (1.0 - avg_bloch_z) / 2.0

    def _solve_qp(self, qp=None, reshape=False):
        """Override parent's _solve_qp to train on batch of training vectors.

        Instead of solving a single quadratic program, this method trains
        the QAOA circuit on all training samples to minimise cross-entropy
        loss. ``qp`` is ignored; ``self.n_features_`` is used directly.

        Parameters
        ----------
        qp : QuadraticProgram or None
            Unused; retained only for interface compatibility with parent.
        reshape : bool, default=False
            Not used in batch training.

        Returns
        -------
        None
            Training results are stored in class attributes.
        """
        if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
            raise ValueError("Training data not set. Call fit() first.")

        n_samples = self.X_train_.shape[0]
        n_var = self.n_features_

        # Build QAOA circuit using shared parent helper
        ansatz_0, continuous_input_params = self._build_ansatz(n_var)

        # Store ansatz and input params for prediction
        self._ansatz = ansatz_0
        self._continuous_input_params = continuous_input_params

        # Separate gamma (cost/mixer) params from theta (input-encoding) params
        theta_set = set(continuous_input_params)
        gamma_params = [p for p in ansatz_0.parameters if p not in theta_set]
        self._gamma_params = gamma_params

        # Training loss history
        self.training_loss_history_ = []

        def loss(params):
            """Cross-entropy loss over all training samples."""
            # Pre-bind gamma params once per loss call (cheaper than full bind × N)
            gamma_dict = {p: v for p, v in zip(gamma_params, params)}
            circuit_with_gamma = ansatz_0.assign_parameters(gamma_dict)

            total_loss = 0.0
            for i in range(n_samples):
                # Rebind only the theta (input) params per sample
                theta_dict = {
                    p: v for p, v in zip(continuous_input_params, self.X_train_[i])
                }
                circuit = circuit_with_gamma.assign_parameters(theta_dict)
                state_vec = Statevector(circuit)

                # Get predicted probability for class 1
                prob_pred = self._extract_class_probability(state_vec, n_var)
                prob_pred = np.clip(prob_pred, 1e-10, 1 - 1e-10)

                # Binary cross-entropy loss
                y_true = self.y_train_[i]
                loss_i = -(
                    y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
                )
                total_loss += loss_i

            avg_loss = total_loss / n_samples
            self.training_loss_history_.append(avg_loss)
            return avg_loss

        # Derive num_params from circuit — robust to mixer parameterisation changes
        num_params = ansatz_0.num_parameters - len(continuous_input_params)
        rng = check_random_state(self.random_state)
        initial_guess = rng.uniform(0, np.pi / 2, num_params)
        bounds = [(0, np.pi)] * num_params

        # Optimize
        print(
            f"Training QAOA classifier with {n_samples} samples, "
            f"{n_var} features, {num_params} circuit parameters..."
        )
        start_time = time.time()

        result = self.optimizer.minimize(loss, initial_guess, bounds=bounds)

        stop_time = time.time()
        self.run_time_ = stop_time - start_time
        self.optim_params_ = result.x

        print(f"Training completed in {self.run_time_:.2f}s")
        print(f"Final loss: {self.training_loss_history_[-1]:.4f}")

        # Store final state vector for interface compatibility (first training sample)
        gamma_dict = {p: v for p, v in zip(gamma_params, self.optim_params_)}
        circuit_partial = ansatz_0.assign_parameters(gamma_dict)
        theta_dict = {
            p: v for p, v in zip(continuous_input_params, self.X_train_[0])
        }
        optimized_circuit = circuit_partial.assign_parameters(theta_dict)
        self.state_vector_ = Statevector(optimized_circuit)
        self.minimum_ = self.training_loss_history_[-1]

    def fit(self, X, y):
        """Fit the QAOA classifier on training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vectors.
        y : ndarray, shape (n_samples,)
            Target labels (binary: 0 or 1).

        Returns
        -------
        self : ContinuousQIOCEClassifier
            Fitted classifier.
        """
        # Store classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Currently only binary classification is supported")

        # Ensure labels are 0 and 1
        self.y_train_ = np.where(y == self.classes_[0], 0, 1)

        # Check feature dimension
        n_samples, n_features = X.shape
        if n_features > self.max_features:
            raise ValueError(
                f"Input has {n_features} features but max_features={self.max_features}. "
                "Apply dimensionality reduction first."
            )

        self.n_features_ = n_features

        # Normalise features and store training statistics in a single pass
        self.X_train_ = self._fit_normalize(X)

        # Train the circuit
        self._solve_qp()

        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test vectors.

        Returns
        -------
        proba : ndarray, shape (n_samples, 2)
            Class probabilities.
        """
        if not hasattr(self, "optim_params_"):
            raise ValueError("Model not trained. Call fit() first.")

        n_samples, _ = X.shape

        # Normalise using training statistics
        X_range = self.X_max_ - self.X_min_
        X_range[X_range == 0] = 1.0
        X_norm = (X - self.X_min_) / X_range * np.pi

        # Pre-bind gamma params once for all test samples
        gamma_dict = {p: v for p, v in zip(self._gamma_params, self.optim_params_)}
        circuit_with_gamma = self._ansatz.assign_parameters(gamma_dict)

        proba = np.zeros((n_samples, 2))
        for i in range(n_samples):
            theta_dict = {
                p: v for p, v in zip(self._continuous_input_params, X_norm[i])
            }
            circuit = circuit_with_gamma.assign_parameters(theta_dict)
            state_vec = Statevector(circuit)

            prob_class_1 = self._extract_class_probability(state_vec, self.n_features_)
            proba[i, 0] = 1.0 - prob_class_1
            proba[i, 1] = prob_class_1

        return proba

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test vectors.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        y_pred_binary = np.argmax(proba, axis=1)
        return self.classes_[y_pred_binary]

    def score(self, X, y):
        """Return accuracy score.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test vectors.
        y : ndarray, shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Made with Bob
