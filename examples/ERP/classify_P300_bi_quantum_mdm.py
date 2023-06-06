"""
====================================================================
Classification of P300 datasets from MOABB using Quantum MDM
====================================================================

The mean and the distance in MDM algorithm are formualted as
optimization problems. These optimization problems are translated
to Qiskit using Docplex and additional glue code.

Classification can be run either on emulation or real quantum computer.

If you want to use GPU, you need to use qiskit-aer-gpu that will replace
qiskit-aer. It is only available on Linux.
pip install qiskit-aer-gpu

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances, Covariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann_qiskit.classification import (
    QuanticSVM,
    QuanticVQC,
    QuanticMDM,
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from operator import itemgetter

from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb import set_log_level

from moabb.datasets import (
    bi2012,
    bi2013a,
    bi2014a,
    bi2014b,
    bi2015a,
    bi2015b,
    BNCI2014008,
    BNCI2014009,
    BNCI2015003,
    EPFLP300,
    Lee2019_ERP,
)

from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.classification import (
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
n_subjects = 5
for dataset in datasets:
    dataset.subject_list = dataset.subject_list[0:n_subjects]

overwrite = True  # set to True if we want to overwrite cached results
pipelines = {}

pipelines["Xdawn+QuanticMDM"] = make_pipeline(
    XdawnCovariances(nfilter=8, estimator="lwf", xdawn_estimator="scm"),
    QuanticMDM(quantum=False),
)

pipelines["XDawnCov+ClassicMDM"] = make_pipeline(XdawnCovariances(4), MDM())

print("Total pipelines to evaluate: ", len(pipelines))

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

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
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, zorder=1, palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
