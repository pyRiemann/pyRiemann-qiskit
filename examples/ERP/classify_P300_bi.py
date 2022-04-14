"""
====================================================================
ERP EEG decoding with Quantum Classifier.
====================================================================

It uses QuantumClassifierWithDefaultRiemannianPipeline. This pipeline uses
Riemannian Geometry and Tangent Space to generate features and a quantum SVM
classifier. It uses MOABB for the evaluation and comparison with other
classifiers.

In QuantumClassifierWithDefaultRiemannianPipeline():
If parameter "shots" is None then a classical SVM is used similar to the one
in scikit learn.
If "shots" is not None and IBM Qunatum token is provided with "q_account_token"
then a real Quantum computer will be used.

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from pyriemann.estimation import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# bi2012,bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014009
from moabb import set_log_level
from moabb.datasets import bi2012 
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.classification import QuantumClassifierWithDefaultRiemannianPipeline
from sklearn.decomposition import PCA
from pyriemann_qiskit.utils.filtering import Vectorizer
from sklearn import svm

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

# bi2012(),bi2013a(),bi2014a(),bi2014b(),bi2015a(),bi2015b(),BNCI2014009()
datasets = [bi2012()]  

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 5
for dataset in datasets:
    dataset.subject_list = dataset.subject_list[0:n_subjects]

overwrite = True # set to True if we want to overwrite cached results

pipelines = {}

# new pipeline provided by pyRiemann-qiskit
# you can choose between classical SVM and Quantum SVM
pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None,  # 'None' forces classic SVM
    nfilter=2,  # default 2
    # default n_components=10, a higher value renders better performance with
    # the SVM version used in qiskit
    dim_red=PCA(n_components=10), 
    #q_account_token='' #IBM Quantum TOKEN
    )

# Here we provide a pipeline for comparison:

# This is a standard pipeline similar to
# QuantumClassifierWithDefaultRiemannianPipeline, but with LDA classifier
# instead.
pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances #applies XDawn and calculates the convariance matrice, output it matrices
    ( 
        nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"
    ),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto"),
)

print ("Total pipelines to evaluate: ", len(pipelines))

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
)

results = evaluation.process(pipelines)

print("Aaveraging the session performance:")
print(results.groupby('pipeline').mean('score')[['score', 'time']])

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
sns.pointplot(data=results, y="score", x="pipeline", 
              ax=ax, zorder=1, 
              palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

fig.show()
