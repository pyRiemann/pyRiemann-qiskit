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
#from moabb.datasets import BNCI2014009
from moabb.datasets import bi2012, bi2013a, bi2014a, bi2014b, bi2015a, bi2015b

from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.classification import QuantumClassifierWithDefaultRiemannianPipeline


##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

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

datasets = [bi2012()] #bi2012, bi2013a, bi2014a, bi2014b, bi2015a, bi2015b

#dataset = bi2012()
#dataset.subject_list = dataset.subject_list[1:2]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}
pipelines["Quantum+Riemannian"] = QuantumClassifierWithDefaultRiemannianPipeline()

#pipelines = { QuantumClassifierWithDefaultRiemannianPipeline() }

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
)

results = evaluation.process(pipelines)