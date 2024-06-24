# -*- coding: utf-8 -*-
"""

A demo on how to use benchmark_alpha.
Performs a benchmark of several variations of the NCH algorithm.

@author: anton andreev
"""

from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from pyriemann.classification import MDM
import os

from pyriemann_qiskit.classification import QuanticNCH
from heavy_benchmark import benchmark_alpha, plot_stat

# start configuration
hb_max_n_subjects = 3
hb_n_jobs = 12
hb_overwrite = False
# end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}

pipelines["NCH+RANDOM_HULL"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    QuanticNCH(
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="random",
        quantum=False,
    ),
)

pipelines["NCH+MIN_HULL"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    QuanticNCH(
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=False,
    ),
)

# this is a non quantum pipeline
pipelines["XD+MDM"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    MDM(),
)

results = benchmark_alpha(
    pipelines,
    max_n_subjects=hb_max_n_subjects,
    n_jobs=hb_n_jobs,
    overwrite=hb_overwrite,
)

print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# save results
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe.csv"
)
results.to_csv(save_path, index=True)

# Provides statistics
plot_stat(results)
