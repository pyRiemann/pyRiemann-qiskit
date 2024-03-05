"""
====================================================================
Classification of P300 datasets from MOABB using Quantum MDM
====================================================================

The mean and the distance in MDM algorithm are formulated as
optimization problems. These optimization problems are translated
to Qiskit using Docplex and additional glue code. These optimizations
are enabled when we use cpm mean or cpm distance. This is set
using the 'metric' parameter of the QuantumMDMWithRiemannianPipeline.

Classification can be run either on emulation or real quantum computer.

If you want to use GPU, you need to use qiskit-aer-gpu that will replace
qiskit-aer. It is only available on Linux.

pip install qiskit-aer-gpu

pip install moabb==0.5.0

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from moabb.datasets import (
    # bi2012,
    # bi2013a,
    # bi2014a,
    # bi2014b,
    # bi2015a,
    # bi2015b,
    # BNCI2014008,
    BNCI2014009,
    # BNCI2015003,
    # EPFLP300,
    # Lee2019_ERP,
)
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300

# inject cpm distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumMDMVotingClassifier,
    QuantumMDMWithRiemannianPipeline,
)

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

paradigm = P300()

# Datasets:
# name, electrodes, subjects
# bi2013a	    16	24 (normal)
# bi2014a    	16	64 (usually low performance)
# BNCI2014009	16	10 (usually high performance)
# BNCI2014008	 8	 8
# BNCI2015003	 8	10
# bi2015a        32  43
# bi2015b        32  44

datasets = [BNCI2014009()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 2
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[0:n_subjects]

# Change this to true to test the quantum optimizer
quantum = False

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

pipelines = {}

pipelines["mean=qlogeuclid/distance=logeuclid"] = QuantumMDMWithRiemannianPipeline(
    metric="mean", quantum=quantum
)

pipelines["mean=logeuclid/distance=qlogeuclid"] = QuantumMDMWithRiemannianPipeline(
    metric="distance", quantum=quantum
)

pipelines["Voting qlogeuclid"] = QuantumMDMVotingClassifier(quantum=quantum)

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
