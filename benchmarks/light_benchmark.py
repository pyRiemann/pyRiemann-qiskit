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
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from sklearn.decomposition import PCA

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
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

paradigm = P300(resample=128)

dataset = bi2012()  # MOABB provides several other P300 datasets

X, y, _ = paradigm.get_data(dataset, subjects=[1])

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
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=2,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto"),  # you can use other classifiers
)

scores = {}

for key, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores[key] = score


print("Scores: ", scores)
