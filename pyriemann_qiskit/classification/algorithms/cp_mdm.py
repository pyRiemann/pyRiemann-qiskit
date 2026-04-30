from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from pyriemann.classification import MDM
from pyriemann.utils.utils import check_metric

from ...utils.distance import distance_functions
from ...utils.docplex import ClassicalOptimizer
from ...utils.mean import mean_functions
from ...utils.utils import is_qfunction


class CpMDM(MDM):
    """Quantum-enhanced MDM classifier

    This class is a constraint programming (CP) implementation of the
    Minimum Distance to Mean (MDM) [1]_, which can run with quantum optimization.
    Only log-Euclidean distance between trial and class prototypes is supported
    at the moment, but any type of metric can be used for centroid estimation.

    Parameters
    ----------
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Notes
    -----
    .. versionadded:: 0.4.2
    .. versionchanged:: 0.6.0
        Moved to algorithms sub-package

    See Also
    --------
    pyriemann.classification.MDM

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    """

    def __init__(self, optimizer=ClassicalOptimizer(), **params):
        self.optimizer = optimizer
        super().__init__(**params)

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray, shape (n_trials,)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials,), default=None
            weights for each trial, not used.

        Returns
        -------
        self : CpMDM instance
            The CpMDM instance.
        """

        self.metric_mean, self.metric_dist = check_metric(self.metric)
        if is_qfunction(self.metric_mean):
            self.classes_ = np.unique(y)

            if sample_weight is None:
                sample_weight = np.ones(X.shape[0])

            mean_func = mean_functions[self.metric_mean]

            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_func)(
                    X[y == c],
                    sample_weight=sample_weight[y == c],
                    optimizer=self.optimizer,
                )
                for c in self.classes_
            )

            self.covmeans_ = np.stack(self.covmeans_, axis=0)

            return self
        else:
            return super().fit(X, y, sample_weight)

    def _predict_distances(self, X):
        if is_qfunction(self.metric_dist):
            distance = distance_functions[self.metric_dist]

            if "hull" in self.metric_dist:
                warn("qdistances to hull should not be use inside MDM")
                weights = [distance(self.covmeans_, x, self.optimizer) for x in X]
            else:
                warn(
                    "q-distances for MDM are toy functions.\
                        Use pyRiemann distances instead."
                )
                weights = [distance(self.covmeans_, x, self.optimizer) for x in X]
            return 1 - np.array(weights)
        else:
            return super()._predict_distances(X)
