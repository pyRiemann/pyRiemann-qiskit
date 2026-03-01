"""Quantum State MDM classifier."""

import math

import numpy as np
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _encode(x_t, n_channels_padded):
    """Normalize a time sample to a unit-norm quantum state (amplitude encoding).

    Parameters
    ----------
    x_t : ndarray, shape (n_channels,)
        A single EEG time sample.
    n_channels_padded : int
        Padded dimension (power of 2).

    Returns
    -------
    psi : ndarray, shape (n_channels_padded,) or None
        Unit-norm state vector, or None if the sample has near-zero norm.
    """
    x_pad = np.zeros(n_channels_padded)
    x_pad[: len(x_t)] = x_t
    norm = np.linalg.norm(x_pad)
    if norm < 1e-12:
        return None
    return x_pad / norm


def _score_trial(x, density_matrices, classes, n_channels_padded, n_channels):
    """Compute per-class fidelity scores for a single trial.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_times)
        Single EEG trial.
    density_matrices : dict
        Per-class density matrices, keyed by label.
    classes : ndarray
        Ordered class labels.
    n_channels_padded : int
    n_channels : int

    Returns
    -------
    scores : ndarray, shape (n_classes,)
        Average quantum fidelity ⟨ψ_t|ρ_c|ψ_t⟩ for each class.
    """
    n_times = x.shape[1]
    scores = np.zeros(len(classes))
    for ci, c in enumerate(classes):
        rho = density_matrices[c]
        F = 0.0
        T_eff = 0
        for t in range(n_times):
            psi = _encode(x[:, t], n_channels_padded)
            if psi is None:
                continue
            psi_trunc = psi[:n_channels]
            F += psi_trunc @ rho @ psi_trunc
            T_eff += 1
        scores[ci] = F / T_eff if T_eff > 0 else 0.0
    return scores


class QuantumStateMDM(BaseEstimator, ClassifierMixin):
    """Quantum state nearest mean classifier.

    Takes raw EEG epochs as input (no covariance estimation step).
    Each time sample is treated as a quantum state via amplitude encoding.
    Classification is by quantum fidelity with per-class density matrices.

    This is the quantum analog of pyriemann's MDM, operating on quantum
    states rather than SPD matrices.

    Parameters
    ----------
    n_jobs : int, default=1
        Number of parallel jobs over trials at predict time.

    Attributes
    ----------
    density_matrices_ : dict[label -> ndarray (n_channels, n_channels)]
        Per-class density matrices fitted from training data.
    classes_ : ndarray
        Unique class labels seen at fit time.
    n_channels_ : int
        Number of EEG channels.
    n_qubits_ : int
        Number of qubits (ceil(log2(n_channels))).
    n_channels_padded_ : int
        Padded channel dimension (2 ** n_qubits_).

    Notes
    -----
    .. versionadded:: 0.6.0
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit per-class density matrices from raw EEG epochs.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Raw EEG epochs.
        y : array-like, shape (n_trials,)
            Class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_channels = X.shape[1]
        n_qubits = math.ceil(math.log2(n_channels)) if n_channels > 1 else 1
        n_channels_padded = 2**n_qubits

        self.n_channels_ = n_channels
        self.n_qubits_ = n_qubits
        self.n_channels_padded_ = n_channels_padded
        self.classes_ = np.unique(y)

        self.density_matrices_ = {}
        for c in self.classes_:
            rho_c = np.zeros((n_channels_padded, n_channels_padded))
            count = 0
            for i in np.where(y == c)[0]:
                n_times = X[i].shape[1]
                for t in range(n_times):
                    psi = _encode(X[i][:, t], n_channels_padded)
                    if psi is None:
                        continue
                    rho_c += np.outer(psi, psi)
                    count += 1
            if count > 0:
                rho_c /= count
            # Truncate back to original channel dimension
            self.density_matrices_[c] = rho_c[:n_channels, :n_channels]

        return self

    def _compute_scores(self, X):
        """Return raw fidelity scores for all trials.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        scores : ndarray, shape (n_trials, n_classes)
        """
        check_is_fitted(self, ["density_matrices_", "classes_"])
        X = np.asarray(X)

        all_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_score_trial)(
                X[i],
                self.density_matrices_,
                self.classes_,
                self.n_channels_padded_,
                self.n_channels_,
            )
            for i in range(len(X))
        )
        return np.array(all_scores)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        y_pred : ndarray, shape (n_trials,)
        """
        scores = self._compute_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities via softmax of fidelity scores.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        proba : ndarray, shape (n_trials, n_classes)
        """
        scores = self._compute_scores(X)
        return softmax(scores, axis=1)
