"""
====================================================================
Brain-Invaders with illiteracy classification example
====================================================================

In this example, we consider the dataset BI_Il which contains
all the subjects from Brain Invaders, having an AUC <= 0.7.

Different pipelines (quantum and classical) are benchmarked.
"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import warnings
from enum import Enum

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mne.decoding import Vectorizer
from moabb import set_log_level
from moabb.datasets.compound_dataset import BI_Il
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from pyriemann_qiskit.ensemble import JudgeClassifier
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
    QuantumMDMVotingClassifier,
    QuantumMDMWithRiemannianPipeline,
)

# inject convex distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa

print(__doc__)


##############################################################################
# getting rid of the warnings about the future
# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore")
set_log_level("info")


##############################################################################
# Initialization
# --------------
#
# 1) Create paradigm
# 2) Load datasets

paradigm = P300()

datasets = [BI_Il()]

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
    xDAWNCov_TsLDA = "(C) XdawnCov+TsLDA"
    xDAWN_LDA = "(C) Xdawn+LDA"
    xDAWNCov_MDM = "(C) XdawnCov+MDM"
    xDAWNCov_TsSVC = "(C) XdawnCov+TsSVC"
    ERPCov_CvxMDM_Dist = "(C) ERPCov+CvxMDM+Dist"
    ERPCov_QMDM_Dist = "(Q) ERPCov+QMDM+Dist"
    xDAWNCov_TsQSVC = "(Q) XdawnCov+TsQSVC"
    Vot_QMDM_Dist_Mean = "(Q) Vot+QMDM+Dist+Mean"
    Vot_QMDM_MDM = "(Q) Vot+QMDM+MDM"
    Judge_QMDM_MDM_TsLDA = "(Q) Judge_QMDM_MDM"


pipelines = {}

# The dataset is particularly long to process.
# When USE_PLACEHOLDERS is True,
# a standard (and fast) Ts+LDA classifier is used.
USE_PLACEHOLDERS = False


def placeholder(key):
    if not USE_PLACEHOLDERS:
        return
    pipelines[key] = Pipeline(
        steps=[
            (
                key,
                XdawnCovariances(
                    nfilter=4,
                    classes=[labels_dict["Target"]],
                    estimator="scm",  # add to classification?
                    xdawn_estimator="lwf",
                ),
            ),
            ("TS", TangentSpace()),
            ("LDA", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )


# Classical Pipelines

pipelines[PIP.xDAWNCov_TsLDA.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",  # add to classification?
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto"),
)
placeholder(PIP.xDAWNCov_TsLDA.value)

pipelines[PIP.xDAWNCov_TsSVC.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    SVC(),
)
placeholder(PIP.xDAWNCov_TsSVC.value)

pipelines[PIP.xDAWNCov_MDM.value] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    MDM(),
)
placeholder(PIP.xDAWNCov_MDM.value)

pipelines[PIP.xDAWN_LDA.value] = make_pipeline(
    Xdawn(nfilter=3),
    Vectorizer(),
    LDA(solver="lsqr", shrinkage="auto"),
)
placeholder(PIP.xDAWN_LDA.value)

pipelines[PIP.ERPCov_CvxMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    metric="distance", quantum=False
)
placeholder(PIP.ERPCov_CvxMDM_Dist.value)

# Quantum Pipelines

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    metric="distance", quantum=True
)
placeholder(PIP.ERPCov_QMDM_Dist.value)

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    metric="distance", quantum=True
)
placeholder(PIP.ERPCov_QMDM_Dist.value)

pipelines[PIP.Vot_QMDM_Dist_Mean.value] = QuantumMDMVotingClassifier(quantum=True)
placeholder(PIP.Vot_QMDM_Dist_Mean.value)

pipelines[PIP.xDAWNCov_TsQSVC.value] = QuantumClassifierWithDefaultRiemannianPipeline(
    # shots=512,
    # nfilter=4,
    # dim_red=PCA(n_components=10),
)
placeholder(PIP.xDAWNCov_TsQSVC.value)

pipelines[PIP.Judge_QMDM_MDM_TsLDA.value] = make_pipeline(
    JudgeClassifier(
        pipelines[PIP.ERPCov_QMDM_Dist],
        pipelines[PIP.xDAWNCov_MDM],
        pipelines[PIP.xDAWNCov_TsLDA],
    )
)
placeholder(PIP.Judge_QMDM_MDM_TsLDA.value)

pipelines[PIP.Vot_QMDM_MDM.value] = VotingClassifier(
    [
        ("QMDM", pipelines[PIP.ERPCov_QMDM_Dist.value]),
        ("MDM ", pipelines[PIP.xDAWNCov_MDM]),
    ],
    voting="soft",
)
placeholder(PIP.Vot_QMDM_MDM.value)


##############################################################################
# Run evaluation
# --------------
#
# Compare the pipeline using a within session evaluation.

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=False, hdf5_path="hdf5"
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])


# ##############################################################################
# Plot Results
# ------------
#
# Here we plot the results to compare pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

order = np.sort(np.unique(results["pipeline"].to_numpy()))

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
plot.axvline(len(order) // 2 - 0.5, ls="--")
sns.pointplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    palette="Set1",
    hue="pipeline",
    hue_order=order,
    order=order,
).set(title="title")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.legend([], [], frameon=False)
plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=45)
plt.show()
