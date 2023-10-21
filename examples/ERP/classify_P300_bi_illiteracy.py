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

from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from moabb.datasets import BNCI2014009
from moabb.datasets.compound_dataset import BI_Il, Cattan2019_VR_Il
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier

# inject convex distance and mean to pyriemann (if not done already)
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumMDMVotingClassifier,
    QuantumMDMWithRiemannianPipeline,
    QuantumClassifierWithDefaultRiemannianPipeline
)

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

datasets = [Cattan2019_VR_Il()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
# n_subjects = 1
# title = "Datasets: "
# for dataset in datasets:
#     title = title + " " + dataset.code
#     dataset.subject_list = dataset.subject_list[0:n_subjects]

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

pipelines["QuantumMDM-Dist"] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)

pipelines["Voting QuantumMDM"] = QuantumMDMVotingClassifier(quantum=True)


pipelines["QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=1024,
    nfilter=2,
    dim_red=PCA(n_components=10),
)

pipelines["LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto")
)

pipelines["MDM"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=4,
        classes=[labels_dict["Target"]],
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    MDM()
)

pipelines["Voting Q+C MDM"] = VotingClassifier(
                [
                    ("Quantum MDM", pipelines["QuantumMDM-Dist"]),
                    ("MDM ", pipelines["MDM"]),
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
    title="title"
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
