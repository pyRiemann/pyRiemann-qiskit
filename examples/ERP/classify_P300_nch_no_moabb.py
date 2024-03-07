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

# Returns a Test dataset that contains an equal amounts of each class
# y should contain only two classes 0 and 1
def SplitEqual(X, y, samples_n, train_samples_n): #samples_n per class
    
    indicesClass1 = []
    indicesClass2 = []
    
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            indicesClass2.append(i)
            
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
    
    X_test_class1 = X[indicesClass1]
    X_test_class2 = X[indicesClass2]
    
    X_test = np.concatenate((X_test_class1,X_test_class2), axis=0)
    
    #remove x_test from X
    X_train = np.delete(X, indicesClass1 + indicesClass2, axis=0)
    
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    y_test = np.concatenate((Y_test_class1,Y_test_class2), axis=0)
    
    #remove y_test from y
    y_train = np.delete(y, indicesClass1 + indicesClass2, axis=0)
    
    if (X_test.shape[0] != 2 * samples_n or y_test.shape[0] != 2 * samples_n):
        raise Exception("Problem with split 1!")
        
    if (X_train.shape[0] + X_test.shape[0] != X.shape[0] or y_train.shape[0] + y_test.shape[0] != y.shape[0]):
        raise Exception("Problem with split 2!")
    
    ####################################################
    X_train_1 = X_train[(y_train == 1)]
    X_train_1 = X_train_1[0:train_samples_n]
    X_train_2 = X_train[(y_train == 0)]
    X_train_2 = X_train_2[0:train_samples_n,:,:]
    
    X_train  = np.concatenate((X_train_1, X_train_2), axis=0)
    
    y_train = np.concatenate((np.ones(train_samples_n, dtype = np.int8), np.zeros(train_samples_n, dtype = np.int8)), axis=0)
    
    return X_train, X_test, y_train, y_test

db = BNCI2014009()  # BNCI2014008()
n_subjects = 4

for subject_i, subject in enumerate(db.subject_list[1:n_subjects]):
    print("Loading subject:", subject)

    X, y, _ = paradigm.get_data(dataset=db, subjects=[subject])
    y = le.fit_transform(y)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.020, random_state=42, stratify=None
    )
    # X_train, X_test, y_train, y_test = SplitEqual(
    #     X, y , 20, 220
    # )

    print("Test Class 1 count:", sum(y_test))
    print("Test Class 2 count:", len(y_test) - sum(y_test))
    
    # train_size = 350
    # X_train = X_train[0:train_size]
    # y_train = y_train[0:train_size]

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

    #score = pipelines["RG+NCH"].fit(X_train, y_train).score(X_test, y_test)
    #print("Classification score - subject, score:", subject_i, " , ", score)
    pipelines["RG+NCH"].fit(X_train, y_train)
    y_pred = pipelines["RG+NCH"].predict(X_test)
    print("Prediction:   ", y_pred)
    print("Ground truth: ", y_test)
    print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
