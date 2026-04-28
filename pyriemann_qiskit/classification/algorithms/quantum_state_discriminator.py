"""Quantum State Discriminator classifier."""

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _build_operator(X):
    """Construct normalized EEG measurement operator from a single trial.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)

    Returns
    -------
    M : ndarray, shape (n_channels, n_channels)
        Normalized EEG covariance (trace=1), acting as a quantum observable.
    """
    M = X @ X.T / X.shape[1]
    tr = np.trace(M)
    if tr < 1e-12:
        n = X.shape[0]
        return np.eye(n) / n
    return M / tr


def _score_trial(X_i, povm, classes):
    """Compute POVM scores trace(Pi_c . M) for one trial.

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, n_times)
    povm : dict[label -> ndarray (n_channels, n_channels)]
    classes : ndarray

    Returns
    -------
    scores : ndarray, shape (n_classes,)
        Valid probabilities: non-negative and summing to 1.
    """
    M = _build_operator(X_i)
    return np.array([np.sum(povm[c] * M) for c in classes])


class QuantumStateDiscriminator(ClassifierMixin, BaseEstimator):
    """Quantum state classifier using the Pretty Good Measurement (PGM).

    The mental state of the user (class A or B) is modeled as a mixed
    quantum state (density matrix) rho_c, estimated from training EEG via
    quantum state tomography. Class priors pi_c are estimated from class
    frequencies in the training set.

    The classifier is a POVM (Positive Operator-Valued Measure) built via
    the Pretty Good Measurement:

        Pi_c = rho_total^{-1/2} (pi_c * rho_c) rho_total^{-1/2}

    where rho_total = sum_c pi_c * rho_c is the prior-weighted average state.

    The POVM satisfies sum_c Pi_c = I, so scores trace(Pi_c . M) are valid
    probabilities (non-negative, summing to 1) directly from the Born rule —
    no softmax needed.

    For two classes with equal priors, this approximates the Helstrom
    measurement (theoretically optimal quantum state discrimination).

    Parameters
    ----------
    n_jobs : int, default=1
        Number of parallel jobs over trials at predict time.

    Attributes
    ----------
    density_matrices_ : dict[label -> ndarray (n_channels, n_channels)]
        Per-class density matrices rho_c (trace=1, PSD), estimated via
        quantum state tomography.
    povm_ : dict[label -> ndarray (n_channels, n_channels)]
        Per-class POVM elements Pi_c satisfying sum_c Pi_c = I.
    priors_ : dict[label -> float]
        Class prior probabilities estimated from training set frequencies.
    classes_ : ndarray
        Unique class labels seen at fit time.
    n_channels_ : int
        Number of EEG channels.

    Notes
    -----
    .. versionadded:: 0.5.0
    .. versionchanged:: 0.6.0
        Moved to algorithms sub-package
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit class density matrices and POVM from raw EEG epochs.

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

        self.n_channels_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_total = len(y)

        # Step 1: quantum state tomography + prior estimation
        density_matrices = {}
        priors = {}
        for c in self.classes_:
            idx = np.where(y == c)[0]
            priors[c] = len(idx) / n_total
            Sigma_c = np.zeros((self.n_channels_, self.n_channels_))
            for i in idx:
                Sigma_c += X[i] @ X[i].T / X[i].shape[1]
            Sigma_c /= len(idx)
            density_matrices[c] = Sigma_c / np.trace(Sigma_c)

        self.priors_ = priors
        self.density_matrices_ = density_matrices

        # Step 2: Pretty Good Measurement
        # rho_total = sum_c pi_c * rho_c  (prior-weighted average state)
        rho_total = sum(priors[c] * density_matrices[c] for c in self.classes_)

        # rho_total^{-1/2} via eigendecomposition (regularized)
        eigenvalues, eigenvectors = eigh(rho_total)
        inv_sqrt_eig = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10))
        rho_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eig) @ eigenvectors.T

        # Pi_c = rho_total^{-1/2} (pi_c * rho_c) rho_total^{-1/2}
        # satisfies sum_c Pi_c = I by construction
        self.povm_ = {
            c: rho_inv_sqrt @ (priors[c] * density_matrices[c]) @ rho_inv_sqrt
            for c in self.classes_
        }

        return self

    def _compute_scores(self, X):
        """Return POVM scores trace(Pi_c . M) for all trials.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        scores : ndarray, shape (n_trials, n_classes)
            Valid probabilities: non-negative and summing to 1.
        """
        check_is_fitted(self, ["povm_", "classes_"])
        X = np.asarray(X)

        all_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_score_trial)(X[i], self.povm_, self.classes_)
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
        """Predict class probabilities.

        POVM scores are valid probabilities by construction (non-negative,
        summing to 1). No softmax is applied.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        proba : ndarray, shape (n_trials, n_classes)
        """
        return self._compute_scores(X)
