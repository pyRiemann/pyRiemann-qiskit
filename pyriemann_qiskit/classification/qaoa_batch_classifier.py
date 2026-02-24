"""QAOA Batch Classifier with Angle Encoding.

This module implements a quantum classifier that trains a single QAOA circuit
on all training vectors simultaneously, rather than optimizing each component
independently.
"""

import time

import numpy as np
from docplex.mp.model import Model
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_optimization.translators import from_docplex_mp
from sklearn.base import ClassifierMixin

from ..utils.docplex import QAOACVAngleOptimizer, create_mixer_rotational_X_gates


class QAOABatchClassifier(QAOACVAngleOptimizer, ClassifierMixin):
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

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_ : int
        Number of features in training data.
    optim_params_ : ndarray
        Optimized circuit parameters (inherited from QAOACVAngleOptimizer).
    training_loss_history_ : list
        Loss values during training.
    X_min_ : ndarray
        Minimum values for feature normalization.
    X_max_ : ndarray
        Maximum values for feature normalization.
    X_train_ : ndarray
        Normalized training data.
    y_train_ : ndarray
        Training labels (binary: 0 or 1).

    Notes
    -----
    .. versionadded:: 0.6.0

    Examples
    --------
    >>> from pyriemann_qiskit.classification.qaoa_batch_classifier import (
    ...     QAOABatchClassifier
    ... )
    >>> clf = QAOABatchClassifier(n_reps=2, max_features=5)
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
    ):
        # Initialize parent QAOACVAngleOptimizer
        super().__init__(
            create_mixer=create_mixer
            if create_mixer is not None
            else create_mixer_rotational_X_gates(0),
            n_reps=n_reps,
            quantum_instance=quantum_instance,
            optimizer=optimizer if optimizer is not None else L_BFGS_B(),
        )
        self.max_features = max_features

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

    def _create_dummy_qp(self, n_features):
        """Create a dummy docplex quadratic program for interface compatibility.

        The parent's _solve_qp expects a QP to extract n_var. We create a
        simple model with n_features continuous variables.
        """
        prob = Model()

        # Create continuous variables (one per feature)
        for i in range(n_features):
            prob.continuous_var(lb=0, ub=1, name=f"x_{i}")

        # Add a dummy objective (will be replaced by classification loss)
        prob.minimize(0)

        return prob

    def _extract_class_probability(self, state_vec, n_features):
        """Extract class probability from quantum state.

        Uses the average Bloch Z-component across all qubits as a
        discriminative feature for binary classification.

        This aggregates the prob() function logic from parent class
        across all qubits.
        """
        bloch_z_sum = 0.0

        for i in range(n_features):
            # Get reduced density matrix for qubit i
            qubits_to_trace = [j for j in range(n_features) if j != i]

            if qubits_to_trace:
                reduced_dm = partial_trace(state_vec, qubits_to_trace)
            else:
                reduced_dm = state_vec.to_operator()

            # Pauli Z expectation value (same as parent's prob() function)
            pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
            dm_matrix = reduced_dm.data
            bloch_z = np.real(np.trace(dm_matrix @ pauli_z))
            bloch_z_sum += bloch_z

        # Average Bloch Z component, map to [0, 1]
        avg_bloch_z = bloch_z_sum / n_features
        prob_class_1 = (1.0 - avg_bloch_z) / 2.0

        return prob_class_1

    def _solve_qp(self, qp, reshape=False):
        """Override parent's _solve_qp to train on batch of training vectors.

        Instead of solving a single quadratic program, this method trains
        the QAOA circuit on all training samples to minimize cross-entropy loss.

        Parameters
        ----------
        qp : QuadraticProgram
            Docplex model used to extract number of variables.
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
        n_var = qp.get_num_vars()

        # Build QAOA circuit using parent's structure
        # Cost operator: Rx, Ry, Rz gates per qubit (same as parent)
        cost = QuantumCircuit(n_var)
        for i in range(n_var):
            param_rx = Parameter(f"γ_rx_{i}")
            cost.rx(param_rx, i)

            param_ry = Parameter(f"γ_ry_{i}")
            cost.ry(param_ry, i)

            param_rz = Parameter(f"γ_rz_{i}")
            cost.rz(param_rz, i)

        cost_op_has_no_parameter = False
        mixer = self.create_mixer(cost.num_qubits, use_params=cost_op_has_no_parameter)

        # Initial state: encode continuous features using Ry rotations (same as parent)
        initial_state = QuantumCircuit(n_var)
        continuous_input_params = []
        for i in range(n_var):
            param_input = Parameter(f"θ_{i}")
            continuous_input_params.append(param_input)
            initial_state.ry(param_input, i)

        # Build QAOA ansatz (same as parent)
        ansatz_0 = QAOAAnsatz(
            cost_operator=cost,
            reps=self.n_reps,
            initial_state=initial_state,
            mixer_operator=mixer,
        ).decompose()

        # Store ansatz for prediction
        self._ansatz = ansatz_0
        self._continuous_input_params = continuous_input_params

        # Training loss history
        self.training_loss_history_ = []

        # Define batch loss function (this is the key difference from parent)
        def loss(params):
            """Cross-entropy loss over all training samples."""
            total_loss = 0.0

            for i in range(n_samples):
                # Combine feature values with circuit parameters
                all_params = np.concatenate([self.X_train_[i], params])

                # Create state vector
                circuit = ansatz_0.assign_parameters(all_params)
                state_vec = Statevector(circuit)

                # Get predicted probability for class 1
                prob_pred = self._extract_class_probability(state_vec, n_var)

                # Clip to avoid log(0)
                prob_pred = np.clip(prob_pred, 1e-10, 1 - 1e-10)

                # Binary cross-entropy loss
                y_true = self.y_train_[i]
                loss_i = -(
                    y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
                )
                total_loss += loss_i

            # Average loss
            avg_loss = total_loss / n_samples
            self.training_loss_history_.append(avg_loss)

            return avg_loss

        # Initialize circuit parameters
        # 3 parameters per qubit per QAOA layer (same as parent)
        num_params = 3 * n_var * self.n_reps
        initial_guess = np.random.uniform(0, np.pi / 2, num_params)
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

        # Store final state vector with optimized parameters (same as parent)
        # Use first training sample as reference
        optimized_circuit = ansatz_0.assign_parameters(
            np.concatenate([self.X_train_[0], self.optim_params_])
        )
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
        self : QAOABatchClassifier
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

        # Normalize and store training features
        self.X_train_ = self._normalize_features(X)
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)

        # Create dummy QP for interface compatibility
        prob = self._create_dummy_qp(n_features)

        # Convert docplex Model to QuadraticProgram
        qp = from_docplex_mp(prob)

        # Train the circuit by calling parent's _solve_qp (which we override)
        self._solve_qp(qp, reshape=False)

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

        # Normalize using training statistics
        X_range = self.X_max_ - self.X_min_
        X_range[X_range == 0] = 1.0
        X_norm = (X - self.X_min_) / X_range * np.pi

        proba = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Combine feature values with optimized circuit parameters
            all_params = np.concatenate([X_norm[i], self.optim_params_])

            # Create state vector
            circuit = self._ansatz.assign_parameters(all_params)
            state_vec = Statevector(circuit)

            # Get probability for class 1
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
