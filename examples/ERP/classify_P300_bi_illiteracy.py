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
import numpy as np
import seaborn as sns
import warnings

from mne.decoding import Vectorizer
from moabb import set_log_level
from moabb.datasets.compound_dataset import Cattan2019_VR_Il
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

# inject convex distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumMDMVotingClassifier,
    QuantumMDMWithRiemannianPipeline,
    QuantumClassifierWithDefaultRiemannianPipeline,
)


print(__doc__)

##############################################################################
# Judge classifier
# ----------------
#
# On this classifier implementation:
#
# "We trained both the quantum and classical algorithms
# on the balanced dataset[...].
# When the two classifiers disagreed on the label of a given transaction
# in the training set, the transaction was noted.
# These transactions, a subset of the training data of the balanced dataset,
# formed an additional dataset on which a metaclassifier was subsequently
# trained" [1]_.


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
# --------------
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
    xDAWNCov_TsLDA = "(C) XdawnCov+TsLDA"
    xDAWN_LDA = "(C) Xdawn+LDA"
    xDAWNCov_MDM = "(C) XdawnCov+MDM"
    xDAWNCov_TsSVC = "(C) XdawnCov+TsSVC"
    ERPCov_CvxMDM_Dist = "(C) ERPCov+CvxMDM+Dist"
    ERPCov_QMDM_Dist = "(Q) ERPCov+QMDM+Dist"
    xDAWNCov_TsQSVC = "(Q) XdawnCov+TsQSVC"
    Vot_QMDM_Dist_Mean = "(Q) Vot+QMDM+Dist+Mean"
    Vot_QMDM_MDM = "(Q) Vot+QMDM+MDM"
    Judge_QMDM_MDM_TsLDA = "(Q) GradBoost_QMDM_MDM"


pipelines = {}

USE_PLACEHOLDERS = False


def placeholder(key):
    if USE_PLACEHOLDERS:
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


## Classical Pipelines

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
    convex_metric="distance", quantum=False
)
placeholder(PIP.ERPCov_CvxMDM_Dist.value)

## Quantum Pipelines

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)
placeholder(PIP.ERPCov_QMDM_Dist.value)

pipelines[PIP.ERPCov_QMDM_Dist.value] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)
placeholder(PIP.ERPCov_QMDM_Dist.value)

pipelines[PIP.Vot_QMDM_Dist_Mean.value] = QuantumMDMVotingClassifier(quantum=True)
placeholder(PIP.Vot_QMDM_Dist_Mean.value)

pipelines[PIP.xDAWNCov_TsQSVC.value] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=512,
    nfilter=4,
    classes=[labels_dict["Target"]],
    dim_red=PCA(n_components=10),
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
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
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

plt.subplots_adjust(bottom=0.3)
plt.xticks(rotation=45)
plt.show()

###############################################################################
# References
# ----------
# .. [1] M. Grossi et al.,
#       ‘Mixed Quantum–Classical Method for Fraud Detection With Quantum
#       Feature Selection’,
#       IEEE Transactions on Quantum Engineering,
#       doi: 10.1109/TQE.2022.3213474.
