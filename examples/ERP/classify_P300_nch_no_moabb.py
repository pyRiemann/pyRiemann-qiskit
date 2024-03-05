# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:26:17 2024

@author: antona
"""

import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from moabb.datasets import (
    bi2013a,
    BNCI2014008,
    BNCI2014009,
    BNCI2015003,
    EPFLP300,
    Lee2019_ERP,
)
from qiskit_optimization.algorithms import ADMMOptimizer, SlsqpOptimizer
from moabb.paradigms import P300
from sklearn.model_selection import train_test_split
from pyriemann_qiskit.classification import QuanticNCH
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import time
from joblib import Parallel, delayed
from multiprocessing import Process
from PIL import Image

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level

from imblearn.under_sampling import NearMiss

set_log_level("CRITICAL")

paradigm = P300()
labels_dict = {"Target": 1, "NonTarget": 0}

le = LabelEncoder()

db = BNCI2014009()  # BNCI2014008()
n_subjects = 1

for subject_i, subject in enumerate(db.subject_list[0:n_subjects]):
    print("Loading subject:", subject)

    X, y, _ = paradigm.get_data(dataset=db, subjects=[subject])
    y = le.fit_transform(y)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    train_size = 40
    X_train = X_train[0: train_size]
    y_train = y_train[0: train_size]

    pipelines = {}

    pipelines["RG+NCH"] = make_pipeline(
        XdawnCovariances(
            nfilter=3,  # increased, might be a problem for quantum
            classes=[labels_dict["Target"]],
            estimator="lwf",
            xdawn_estimator="scm",
        ),
        QuanticNCH(classical_optimizer=SlsqpOptimizer()),
    )

    score = pipelines["RG+NCH"].fit(X_train, y_train).score(X_test, y_test)
    print("Classification score - subject, score:", subject, score)
