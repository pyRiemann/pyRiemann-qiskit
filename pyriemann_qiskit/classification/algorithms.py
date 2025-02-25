"""
Contains the base class for all quantum classifiers
as well as several quantum classifiers than can run
in several modes quantum/classical and simulated/real
quantum computer.
"""
import logging
import random

import numpy as np
from joblib import Parallel, delayed
from pyriemann.utils.distance import distance
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from ..utils.distance import qdistance_logeuclid_to_convex_hull
from ..utils.docplex import get_global_optimizer, set_global_optimizer

logging.basicConfig(level=logging.WARNING)


class NearestConvexHull(ClassifierMixin, TransformerMixin, BaseEstimator):

    """Classification by Nearest Convex Hull (NCH).

    In Nearest Convex Hull (NCH) classifier [1]_, each class is modelized by
    the convex hull generated by the matrices corresponding to this class.
    There is no training. Calculating a distance to a hull is an optimization
    problem and it is calculated for each testing SPD matrix and each hull.
    The minimal distance defines the predicted class.

    Notes
    -----
    .. versionadded:: 0.2.0

    Parameters
    ----------
    n_jobs : int, default=6
        The number of jobs to use for the computation. This works by computing
        each of the hulls in parallel.
    n_hulls_per_class : int, default=3
        The number of hulls used per class, when subsampling is "random".
    n_samples_per_hull : int, default=15
        Defines how many matrices are used to build a hull.
        -1 will include all matrices per class.
        If subsampling is "full", this parameter is defaulted to -1.
    subsampling : {"min", "random", "full"}, default="min"
        Subsampling strategy of training set to estimate distance to hulls:

        - "full" computes the hull on the entire training matrices, as in [1]_;
        - "min" estimates hull using the n_samples_per_hull closest matrices;
        - "random" estimates hull using n_samples_per_hull random matrices.
    seed : float, default=None
        Optional random seed to use when subsampling is set to "random".

    References
    ----------
    .. [1] `Convex Class Model on Symmetric Positive Definite Manifolds
        <https://arxiv.org/pdf/1806.05343>`_
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        Image and Vision Computing, 2019.
    """

    def __init__(
        self,
        n_jobs=6,
        n_hulls_per_class=3,
        n_samples_per_hull=10,
        subsampling="min",
        seed=None,
    ):
        """Init."""
        self.n_jobs = n_jobs
        self.n_samples_per_hull = n_samples_per_hull
        self.n_hulls_per_class = n_hulls_per_class
        self.matrices_per_class_ = {}
        self.debug = False
        self.subsampling = subsampling
        self.seed = seed

        if subsampling not in ["min", "random", "full"]:
            raise ValueError(f"Unknown subsampling type {subsampling}.")

        if subsampling == "full":
            # From code perspective, "full" strategy is the same as min strategy
            # without sorting
            self.n_samples_per_hull = -1

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        self : NearestConvexHull instance
            The NearestConvexHull instance.
        """

        self.random_generator = random.Random(self.seed)

        if self.debug:
            print("Start NCH Train")
        self.classes_ = np.unique(y)

        for c in self.classes_:
            self.matrices_per_class_[c] = X[y == c]

        if self.debug:
            print("Samples per class:")
            for c in self.classes_:
                print("Class: ", c, " Count: ", self.matrices_per_class_[c].shape[0])

            print("End NCH Train")

    def _process_sample_min_hull(self, x):
        """Finds the closest matrices and uses them to build a single hull per class"""
        dists = []

        for c in self.classes_:
            dist = distance(self.matrices_per_class_[c], x, metric="logeuclid")[:, 0]
            # take the closest matrices
            indexes = np.argsort(dist)[0 : self.n_samples_per_hull]

            if self.debug:
                print("Distances to test sample: ", dist)
                print("Smallest N distances indexes:", indexes)
                print("Smallest N distances: ")
                for pp in indexes:
                    print(dist[pp])

            d = qdistance_logeuclid_to_convex_hull(
                self.matrices_per_class_[c][indexes], x
            )

            if self.debug:
                print("Final hull distance:", d)

            dists.append(d)

        return dists

    def _process_sample_random_hull(self, x):
        """Uses random matrices to build a hull, can be several hulls per class"""
        dists = []

        for c in self.classes_:
            dist_total = 0

            # using multiple hulls
            for i in range(0, self.n_hulls_per_class):
                if self.n_samples_per_hull == -1:  # use all data per class
                    hull_data = self.matrices_per_class_[c]
                else:  # use a subset of the data per class
                    random_samples = self.random_generator.sample(
                        range(self.matrices_per_class_[c].shape[0]),
                        k=self.n_samples_per_hull,
                    )
                    hull_data = self.matrices_per_class_[c][random_samples]

                dist = qdistance_logeuclid_to_convex_hull(hull_data, x)
                dist_total = dist_total + dist

            dists.append(dist_total)

        return dists

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        dists = []

        if self.debug:
            print("Total test samples:", X.shape[0])

        if self.subsampling == "min" or self.subsampling == "full":
            self._process_sample = self._process_sample_min_hull
        elif self.subsampling == "random":
            self._process_sample = self._process_sample_random_hull
        else:
            raise ValueError(f"Unknown subsampling type {self.subsampling}.")

        parallel = self.n_jobs > 1

        if self.debug:
            if parallel:
                print("Running in parallel")
            else:
                print("Not running in parallel")

        if parallel:
            # Get global optimizer in this process
            optimizer = get_global_optimizer(default=None)

            def job(x):
                # Set the global optimizer inside the new process
                set_global_optimizer(optimizer)
                return self._process_sample(x)

            dists = Parallel(n_jobs=self.n_jobs)(delayed(job)(x) for x in X)

        else:
            for x in X:
                dist = self._process_sample(x)
                dists.append(dist)

        dists = np.asarray(dists)

        return dists

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest convex hull.
        """
        if self.debug:
            print("Start NCH Predict")
        dist = self._predict_distances(X)

        predictions = self.classes_[dist.argmin(axis=1)]

        if self.debug:
            print("End NCH Predict")

        return predictions

    def transform(self, X):
        """Get the distance to each convex hull.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The distance to each convex hull.
        """

        if self.debug:
            print("NCH Transform")
        return self._predict_distances(X)
