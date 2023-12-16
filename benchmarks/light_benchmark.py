"""
====================================================================
Light Benchmark
====================================================================

This benchmark is a non-regression performance test, intended
to run on Ci with each PRs.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_P300_bi.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.paradigms import P300
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from sklearn.decomposition import PCA
import warnings
import os

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

##############################################################################

paradigm = P300(resample=128)

dataset = bi2012()  # MOABB provides several other P300 datasets

X, y, _ = paradigm.get_data(dataset, subjects=[1])

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

pipelines = {}

pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=512,
    nfilter=2,
    dim_red=PCA(n_components=5),
)

pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto"),
)

scores = {}
mean_score = 0

for key, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores[key] = score
    mean_score += score

mean_score /= len(pipelines)

print("Scores: ", scores)

pr_score = os.getenv("PR_SCORE")
if pr_score is None:
    os.setenv("PR_SCORE", mean_score)
else:
    # mean_score is score in main branch
    os.setenv("SUCCESS", pr_score >= mean_score)
