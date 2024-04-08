# -*- coding: utf-8 -*-
"""

Performs a benchmark of several variations of the NCH algorithm.

@author: anton andreev
"""

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline

# from matplotlib import pyplot as plt
# import warnings
# import seaborn as sns
# import pandas as pd
# from moabb import set_log_level
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)

# from moabb.paradigms import P300, MotorImagery, LeftRightImagery
from pyriemann.classification import MDM

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn import svm
# import moabb.analysis.plotting as moabb_plt
# from moabb.analysis.meta_analysis import (
#     compute_dataset_statistics,
#     find_significant_differences,
# )
import os

from pyriemann_qiskit.classification import QuanticNCH
from heavy_benchmark import benchmark_alpha, plot_stat

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

results = benchmark_alpha(pipelines, max_n_subjects=3)

print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# save results
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe.csv"
)
results.to_csv(save_path, index=True)

plot_stat(results)
