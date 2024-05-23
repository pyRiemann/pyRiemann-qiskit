"""
===============================================================================
Quantum autoencoder for signal denoising
===============================================================================
This is a basic example of a quantum autoencoder for signal denoising,
based on [1]_ and [2]_.

There is no particular advantage in using QNN for such a task.

This is experimental and should be used for research purpose only.

"""

# Authors: A. Mostafa, Y. Chauhan, W. Ahmed, and G. Cattan
# License: BSD (3-clause)

import logging
import warnings

from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import Hinss2021
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from qiskit_algorithms.optimizers import COBYLA
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.autoencoders import BasicQnnAutoencoder
from pyriemann_qiskit.utils.filtering import EpochChannelSelection
from pyriemann_qiskit.utils.preprocessing import Vectorizer, Devectorizer

print(__doc__)

##############################################################################
# getting rid of the warnings about the future

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Setting logger level to info, to print the autoencoder trace.

logging.getLogger("root").setLevel(logging.INFO)

##############################################################################
# Initialization
# ----------------
#
# 1) Create paradigm
# 2) Load datasets

events = dict(easy=2, medium=3)
paradigm = RestingStateToP300Adapter(events=events, tmin=0, tmax=0.5)

datasets = [Hinss2021()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
start_subject, stop_subject = 14, 15
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

##############################################################################
# Define a callback to keep trace of the computed costs.

costs = {}


def fn_callback(iter, cost):
    if iter in costs:
        costs[iter].append(cost)
    else:
        costs[iter] = [cost]


##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

pipelines = {}

# An important limitation is that:
# n_components x n_times = 2 ** (num_latent + num_trash)
n_components, n_times = 8, 64

pipelines["QNN+LDA"] = make_pipeline(
    EpochChannelSelection(n_chan=n_components),
    Vectorizer(),
    # Use COBYLA with only 1 iteration (this is for runnin in Ci/Cd)
    BasicQnnAutoencoder(
        num_latent=n_components,
        num_trash=1,
        opt=COBYLA(maxiter=10),
        callback=fn_callback,
    ),
    Devectorizer(n_components, n_times),
    Covariances(),
    TangentSpace(),
    LDA(),
)

pipelines["LDA"] = make_pipeline(
    EpochChannelSelection(n_chan=n_components),
    Vectorizer(),
    Devectorizer(n_components, n_times),
    Covariances(),
    TangentSpace(),
    LDA(),
)

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a cross-sessions evaluation.

evaluation = CrossSessionEvaluation(
    paradigm=paradigm, datasets=datasets, overwrite=True, n_jobs=-1
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

##############################################################################
# Plot the mean cost function

x = []
y = []
for iter in costs.keys():
    x.append(iter)
    c = costs[iter]
    y.append(sum(c) / len(c))

plt.plot(x, y)
plt.xlabel("N of cost evaluation")
plt.ylabel("Cost")
plt.title("Autoencoder Cost")
plt.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] \
#   https://qiskit-community.github.io/qiskit-machine-learning/tutorials/12_quantum_autoencoder.html
# .. [2] A. Mostafa et al., 2024
#   ‘Quantum Denoising in the Realm of Brain-Computer Interfaces: A Preliminary Study’,
#   https://hal.science/hal-04501908
