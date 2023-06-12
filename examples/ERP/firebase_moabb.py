"""
====================================================================
Classification of P300 datasets from MOABB with cache
====================================================================

It demonstrates the use of firebase cache for MOABB with
the QuantumClassifierWithDefaultRiemannianPipeline(). This
pipeline uses Riemannian Geometry, Tangent Space and a quantum SVM
classifier. MOABB is used to access many EEG datasets and also for the
evaluation and comparison with other classifiers.

In QuantumClassifierWithDefaultRiemannianPipeline():
If parameter "shots" is None then a classical SVM is used similar to the one
in scikit learn.
If "shots" is not None and IBM Qunatum token is provided with "q_account_token"
then a real Quantum computer will be used.
You also need to adjust the "n_components" in the PCA procedure to the number
of qubits supported by the real quantum computer you are going to use.
A list of real quantum  computers is available in your IBM quantum account.

"""
# Author: Anton Andreev, Gregoire Cattan
# Modified from plot_classify_P300_bi.py
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.utils import (
    generate_caches,
    filter_subjects_by_incomplete_results,
    add_moabb_dataframe_results_to_caches,
    convert_caches_to_dataframes,
)
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb import set_log_level
from moabb.datasets import bi2012
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

paradigm = P300(resample=128)

datasets = [bi2012()]  # MOABB provides several other P300 datasets
copy_datasets = [bi2012()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 5
for dataset, copy_dataset in zip(datasets, copy_datasets):
    dataset.subject_list = dataset.subject_list[0:n_subjects]
    copy_dataset.subject_list = copy_dataset.subject_list[0:n_subjects]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

# A Riemannian Quantum pipeline provided by pyRiemann-qiskit
# You can choose between classical SVM and Quantum SVM.
pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None,  # 'None' forces classic SVM
    nfilter=2,  # default 2
    # default n_components=10, a higher value renders better performance with
    # the non-qunatum SVM version used in qiskit
    # On a real Quantum computer (n_components = qubits)
    dim_red=PCA(n_components=5),
    # params={'q_account_token': '<IBM Quantum TOKEN>'}
)

# Here we provide a pipeline for comparison:

# This is a standard pipeline similar to
# QuantumClassifierWithDefaultRiemannianPipeline, but with LDA classifier
# instead.
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

# We cache the results on Firebase.
# But you can skip all cache functions bellow if you want.
caches = generate_caches(datasets, pipelines)

# This method remove a subject in a dataset if we already have evaluated
# all pipelines for this subject.
# Therefore we will use a copy of the original datasets.
filter_subjects_by_incomplete_results(caches, copy_datasets, pipelines)

print("Total pipelines to evaluate: ", len(pipelines))
print(
    "Subjects to evaluate",
    sum([len(dataset.subject_list) for dataset in copy_datasets]),
)
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=copy_datasets, suffix="examples", overwrite=overwrite
)

try:
    results = evaluation.process(pipelines)
    add_moabb_dataframe_results_to_caches(results, copy_datasets, pipelines, caches)
except ValueError:
    print("No subjects left to evaluate.")

df = convert_caches_to_dataframes(caches, datasets, pipelines)

print("Averaging the session performance:")
print(df.groupby("pipeline").mean("score")[["score", "time"]])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=df,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)

sns.pointplot(data=df, y="score", x="pipeline", ax=ax, palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
