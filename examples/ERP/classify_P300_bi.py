"""
====================================================================
ERP EEG decoding with Quantum Classifier.
====================================================================

It uses QuantumClassifierWithDefaultRiemannianPipeline on a number of
datasets recorded using the BCI game Brain Invaders.

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticSVM
from pyriemann_qiskit.utils.filtering import NaiveDimRed
from pyriemann_qiskit.datasets import get_mne_sample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score)
from matplotlib import pyplot as plt

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyriemann.estimation import Xdawn, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014009
from moabb.datasets import bi2012, bi2013a, bi2014a, bi2014b, bi2015a, bi2015b

from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.classification import QuantumClassifierWithDefaultRiemannianPipeline


##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

moabb.set_log_level("info")

##############################################################################
# This is an auxiliary transformer that allows one to vectorize data
# structures in a pipeline For instance, in the case of an X with dimensions
# Nt x Nc x Ns, one might be interested in a new data structure with
# dimensions Nt x (Nc.Ns)


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))


##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.


#pipelines = {}

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
#labels_dict = {"Target": 1, "NonTarget": 0}

paradigm = P300(resample=128)

datasets = [bi2012()] #bi2012(), bi2013a(), bi2014a(), bi2014b(), bi2015a(), bi2015b(), BNCI2014009()

#reduce the number of subjects
# nsubjects = 10
# for dataset in datasets:
#     dataset.subject_list = dataset.subject_list[0:nsubjects]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

#new experimental pipeline
from sklearn.decomposition import PCA

pipelines["RG+Quantum"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None, #'None' forces classic SVM
    #gamma = 0.05,
    nfilter=2, 
    dim_red=PCA(n_components=10)
    )

labels_dict = {"Target": 1, "NonTarget": 0}

#standard pipeline 1
pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"
    ),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto"),
)

#standard pipeline 2
pipelines["Xdw+LDA"] = make_pipeline(
    Xdawn(nfilter=2, estimator="scm"), 
    Vectorizer(), 
    LDA(solver="lsqr", shrinkage="auto")
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
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, zorder=1, palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

fig.show()