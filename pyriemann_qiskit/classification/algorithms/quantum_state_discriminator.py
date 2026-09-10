"""Quantum State Discriminator classifier."""

import numpy as np
from pyriemann.estimation import Covariances, TimeDelayCovariances, XdawnCovariances
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


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

    The operator M representing a trial is a trace-normalized covariance
    matrix, estimated by pyriemann rather than here: ``covariance`` selects
    which pyriemann estimator to delegate to.

    Parameters
    ----------
    covariance : {"cov", "erp", "hankel"}, default="cov"
        Which pyriemann estimator builds the operator of a trial.

        - "cov": :class:`pyriemann.estimation.Covariances`, the temporal
          covariance X.X^T. It is the state of a sustained (oscillatory)
          process, so it suits resting-state paradigms, where the classes
          differ in spatial covariance and band power.
        - "erp": :class:`pyriemann.estimation.XdawnCovariances`, which
          concatenates the class-average evoked responses to each
          Xdawn-filtered trial before the covariance, so the operator also
          carries the trial/prototype cross-correlation. Required for
          time-locked paradigms (P300 and other ERPs): a plain covariance
          integrates over time and averages the evoked deflection away.
        - "hankel": :class:`pyriemann.estimation.TimeDelayCovariances`,
          which concatenates ``delays`` time-shifted copies of the trial,
          so the operator carries temporal auto-correlation. Unlike "erp"
          this needs no labels.
    estimator : string, default="scm"
        Covariance estimator passed to the pyriemann estimator above, see
        :func:`pyriemann.geometry.covariance.covariances`.
    nfilter : int, default=8
        Number of Xdawn spatial filters, when ``covariance="erp"``.
        Capped at n_channels by Xdawn.
    prototype_weight : float | None, default=None
        Energy of the prototype block relative to the average trial, when
        ``covariance="erp"``. ``None`` uses the prototypes unscaled.

        The operator is trace-normalized, so its blocks compete for a fixed
        budget, and only the prototype/trial cross-block is discriminative:
        the prototype block is identical for every trial. A class-average
        prototype carries far less energy than a single trial, so without
        boosting it the cross-block claims little of the trace. This matters
        because QSD compares against class-mean density matrices and cannot
        down-weight uninformative entries the way a trained classifier does.

        Applied as a congruence on the covariance, which is exactly
        equivalent to scaling the prototype before estimating it.
    delays : int, default=4
        Number of time-shifted copies, when ``covariance="hankel"``.
    xdawn_estimator : string, default="oas"
        Covariance estimator used to fit Xdawn, when ``covariance="erp"``.
    n_jobs : int, default=1
        Unused, kept for backward compatibility. Scoring is vectorized over
        trials and runs in BLAS, so spreading trials over worker processes
        cost more than it saved.

    Attributes
    ----------
    cov_estimator_ : object
        The fitted pyriemann estimator building the operators.
    density_matrices_ : dict[label -> ndarray (n_channels_, n_channels_)]
        Per-class density matrices rho_c (trace=1, PSD), estimated via
        quantum state tomography.
    povm_ : dict[label -> ndarray (n_channels_, n_channels_)]
        Per-class POVM elements Pi_c satisfying sum_c Pi_c = I.
    priors_ : dict[label -> float]
        Class prior probabilities estimated from training set frequencies.
    classes_ : ndarray
        Unique class labels seen at fit time.
    n_channels_ : int
        Dimension of the density matrices. Equal to the number of EEG
        channels when ``covariance="cov"``, larger otherwise.
    prototype_scale_ : float
        Amplitude applied to the prototype block, when
        ``covariance="erp"`` and ``prototype_weight`` is set.

    Notes
    -----
    .. versionadded:: 0.5.0
    .. versionchanged:: 0.6.0
        Moved to algorithms sub-package
    .. versionchanged:: 0.6.0
        Add ``covariance`` to support time-locked (ERP) paradigms
    .. versionchanged:: 0.6.0
        Delegate covariance estimation to pyriemann; ``n_jobs`` is now
        unused
    """

    def __init__(
        self,
        covariance="cov",
        estimator="scm",
        nfilter=8,
        prototype_weight=None,
        delays=4,
        xdawn_estimator="oas",
        n_jobs=1,
    ):
        self.covariance = covariance
        self.estimator = estimator
        self.nfilter = nfilter
        self.prototype_weight = prototype_weight
        self.delays = delays
        self.xdawn_estimator = xdawn_estimator
        self.n_jobs = n_jobs

    def _make_cov_estimator(self):
        if self.covariance == "cov":
            return Covariances(estimator=self.estimator)
        if self.covariance == "erp":
            return XdawnCovariances(
                nfilter=self.nfilter,
                estimator=self.estimator,
                xdawn_estimator=self.xdawn_estimator,
            )
        if self.covariance == "hankel":
            return TimeDelayCovariances(delays=self.delays, estimator=self.estimator)
        raise ValueError(
            'covariance must be "cov", "erp" or "hankel", got ' f"{self.covariance}"
        )

    def _weight_prototype(self, covmats):
        """Rescale the prototype block of the ERP operator.

        Scaling the prototype rows by s before estimating the covariance
        multiplies the prototype block by s^2 and the cross-blocks by s,
        which is the congruence D.C.D with D = diag(s I_p, I_x).
        """
        if self.covariance != "erp" or self.prototype_weight is None:
            self.prototype_scale_ = 1.0
            return covmats

        n_proto = self.cov_estimator_.P_.shape[0]
        idx = np.arange(covmats.shape[-1])
        proto_energy = np.trace(
            covmats[0, :n_proto, :n_proto]  # identical for every trial
        )
        trial_energy = np.mean(
            np.trace(covmats[:, n_proto:, n_proto:], axis1=-2, axis2=-1)
        )
        self.prototype_scale_ = float(
            np.sqrt(self.prototype_weight * trial_energy / proto_energy)
        )

        scale = np.where(idx < n_proto, self.prototype_scale_, 1.0)
        return covmats * scale[None, :, None] * scale[None, None, :]

    def _operators(self, X, y=None):
        """Trial operators, estimated by pyriemann and prototype-weighted."""
        if y is None:
            covmats = self.cov_estimator_.transform(np.asarray(X))
            if self.covariance == "erp" and self.prototype_weight is not None:
                n_proto = self.cov_estimator_.P_.shape[0]
                idx = np.arange(covmats.shape[-1])
                scale = np.where(idx < n_proto, self.prototype_scale_, 1.0)
                covmats = covmats * scale[None, :, None] * scale[None, None, :]
            return covmats

        self.cov_estimator_ = self._make_cov_estimator()
        covmats = self.cov_estimator_.fit_transform(np.asarray(X), y)
        return self._weight_prototype(covmats)

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
        y = np.asarray(y)
        covmats = self._operators(X, y)

        self.n_channels_ = covmats.shape[-1]
        self.classes_ = np.unique(y)
        n_total = len(y)

        # Step 1: quantum state tomography + prior estimation
        density_matrices, priors = {}, {}
        for c in self.classes_:
            idx = np.where(y == c)[0]
            priors[c] = len(idx) / n_total
            Sigma_c = covmats[idx].mean(axis=0)
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
        self.povm_ = {
            c: rho_inv_sqrt @ (priors[c] * density_matrices[c]) @ rho_inv_sqrt
            for c in self.classes_
        }

        # The PGM sums to the projector onto the support of rho_total, which
        # is the identity only when rho_total is full rank. A rank-deficient
        # rho_total (fewer time samples than channels, or an augmented
        # operator) otherwise leaves sum_c Pi_c != I, and predict_proba would
        # not sum to 1. Share the residual equally between classes: this
        # restores completeness while staying PSD, and stays uninformative on
        # the null space, where no class was observed.
        residual = np.eye(self.n_channels_) - sum(self.povm_.values())
        residual = (residual + residual.T) / 2
        for c in self.classes_:
            self.povm_[c] = self.povm_[c] + residual / len(self.classes_)

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
        covmats = self._operators(X)

        # M = C / trace(C), so trace(Pi_c . M) = sum(Pi_c * C) / trace(C)
        povm = np.stack([self.povm_[c] for c in self.classes_])
        scores = np.einsum("nij,cij->nc", covmats, povm)

        energy = np.trace(covmats, axis1=-2, axis2=-1)
        degenerate = energy < 1e-12
        scores /= np.where(degenerate, 1.0, energy)[:, None]

        if degenerate.any():
            # A silent trial carries no state: M = I / n_channels.
            scores[degenerate] = np.trace(povm, axis1=-2, axis2=-1) / self.n_channels_

        return scores

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
