"""
====================================================================
Classification of P300 datasets from MOABB using Quantum MDM
====================================================================

The mean and the distance in MDM algorithm are formulated as
optimization problems. These optimization problems are translated
to Qiskit using Docplex and additional glue code. These optimizations
are enabled when we use convex mean or convex distance. This is set
using the 'convex_metric' parameter of the QuantumMDMWithRiemannianPipeline.

Classification can be run either on emulation or real quantum computer.

If you want to use GPU, you need to use qiskit-aer-gpu that will replace
qiskit-aer. It is only available on Linux.

pip install qiskit-aer-gpu

pip install moabb==0.5.0

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from enum import Enum
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from moabb.datasets import BNCI2014009
from moabb.datasets.compound_dataset import BI_Il, Cattan2019_VR_Il
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.base import ClassifierMixin
from mne.decoding import Vectorizer

# inject convex distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumMDMVotingClassifier,
    QuantumMDMWithRiemannianPipeline,
    QuantumClassifierWithDefaultRiemannianPipeline,
)

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

import numpy as np

print(__doc__)

########################################## Judge classifier


# Grossi et al
class JudgeClassifier(ClassifierMixin):
    def __init__(self, c1, c2, judge):
        self.c1 = c1
        self.c2 = c2
        self.judge = judge

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y1 = self.c1.fit(X, y).predict(X)
        y2 = self.c2.fit(X, y).predict(X)
        mask = np.not_equal(y1, y2)
        if mask.all() == False:
            self.judge.fit(X, y)
        else:
            y_diff = y[mask]
            X_diff = X[mask]
            self.judge.fit(X_diff, y_diff)

    def predict(self, X):
        y1 = self.c1.predict(X)
        y2 = self.c2.predict(X)
        y_pred = y1
        mask = np.not_equal(y1, y2)
        if mask.all() == False:
            return y_pred
        X_diff = X[mask]
        y_pred[mask] = self.judge.predict(X_diff)
        return y_pred

    def predict_proba(self, X):
        y1_proba = self.c1.predict_proba(X)
        y2_proba = self.c2.predict_proba(X)
        y1 = self.c1.predict(X)
        y2 = self.c2.predict(X)
        predict_proba = (y1_proba + y2_proba) / 2
        mask = np.not_equal(y1, y2)
        if mask.all() == False:
            return predict_proba
        X_diff = X[mask]
        predict_proba[mask] = self.judge.predict_proba(X_diff)
        return predict_proba


##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Initialization
# ----------------
#
# 1) Create paradigm
# 2) Load datasets

paradigm = P300()

datasets = [Cattan2019_VR_Il()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 1
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[0:n_subjects]

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.


class PIP(str, Enum):
    xDAWN_Cov_TsQSVC = "xDAWN+Cov+TsQSVC"
    xDAWN_Cov_TsLDA = "xDAWN+Cov+TsLDA"
    xDAWN_LDA = "xDAWN+LDA"
    xDAWN_Cov_MDM = "xDAWN+Cov+MDM"
    xDAWN_Cov_TsGradBoost = "xDAWN+Cov+TsGradBoost"
    xDAWN_Cov_TsSVC = "xDAWN+Cov+TsSVC"
    ERPCov_QMDM_Dist = "ERPCov+QMDM+Dist"
    ERPCov_CvxMDM_Dist = "ERPCov+CvxMDM+Dist"
    Vot_QMDM_Dist_Mean = "Vot+QMDM+Dist+Mean"
    Vot_QMDM_MDM = "Vot+QMDM+MDM"
    GradBoost_ERPCov_QMDM = "XGBoost_ERPCov_QMDM"


pipelines = {}

## Classical Pipelines

pipelines[PIP.xDAWN_Cov_TsLDA.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",  # add to classification?
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto"),
)

pipelines[PIP.xDAWN_Cov_TsGradBoost.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    GradientBoostingClassifier(),
)

pipelines[PIP.xDAWN_Cov_TsSVC.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    SVC(),
)

pipelines[PIP.xDAWN_Cov_MDM.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    MDM(),
)

pipelines[PIP.xDAWN_LDA.value] = make_pipeline(
    Xdawn(nfilter=3),
    Vectorizer(),
    LDA(solver="lsqr", shrinkage="auto"),
)

pipelines[PIP.ERPCov_CvxMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=False
)

## Quantum Pipelines

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)

pipelines[PIP.Vot_QMDM_Dist_Mean.value] = QuantumMDMVotingClassifier(quantum=True)

pipelines[PIP.xDAWN_Cov_TsQSVC.value] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=1024,
    nfilter=2,
    classes=[labels_dict["Target"]],
    dim_red=PCA(n_components=10),
)

pipelines[PIP.GradBoost_ERPCov_QMDM.value] = make_pipeline(
    JudgeClassifier(
        pipelines[PIP.xDAWN_Cov_TsQSVC],
        pipelines[PIP.xDAWN_Cov_TsLDA],
        pipelines[PIP.xDAWN_Cov_TsGradBoost],
    )
)

pipelines[PIP.Vot_QMDM_MDM.value] = VotingClassifier(
    [
        ("QMDM", pipelines[PIP.ERPCov_QMDM_Dist.value]),
        ("MDM ", pipelines[PIP.xDAWN_Cov_MDM]),
    ],
    voting="soft",
)


##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a within session evaluation.

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])


# ##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

order = np.unique(results["pipeline"].to_numpy())
print(order)

plot = sns.stripplot(
    data=results,
    x="pipeline",
    y="score",
    order=order,
    hue="pipeline",
    hue_order=order,
    jitter=True,
    alpha=0.5,
    palette="Set1",
)
plot.axvline((len(order) - 1) // 2)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title="title"
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
