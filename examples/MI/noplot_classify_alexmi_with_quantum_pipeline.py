"""
============================================================================
Classification of MI datasets from MOABB using MDM and quantum-enhanced MDM
============================================================================

This example demonstrates how to use quantum pipeline on a MI dataset.

pip install moabb==0.5.0

"""
# Author: Gregoire Cattan
# Modified from ERP/classify_P300_bi_quantum_mdm.py
# License: BSD (3-clause)

import warnings

import seaborn as sns
from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import AlexMI
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery

from pyriemann_qiskit.pipelines import QuantumMDMWithRiemannianPipeline

# inject cpm distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa

from helpers.alias import ERPCov_MDM

print(__doc__)

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

paradigm = MotorImagery(events=["feet", "right_hand"], n_classes=2)

datasets = [AlexMI()]

# reduce the number of subjects
n_subjects = 2
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[0:n_subjects]

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

pipelines = {}

# Will run QAOA under the hood
pipelines["mean=logeuclid/distance=cpm"] = QuantumMDMWithRiemannianPipeline(
    metric="distance", quantum=True
)

# Classical baseline for evaluation
pipelines["R-MDM"] = ERPCov_MDM

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipelines using a within session evaluation.

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])


# ##############################################################################
# # Plot Results
# # ----------------
# #
# # Here we plot the results to compare two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title=title
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
