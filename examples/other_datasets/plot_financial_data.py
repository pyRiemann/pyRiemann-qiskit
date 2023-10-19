"""
====================================================================
Suspicious financial activity detection using quantum computer
====================================================================

In this example, we will illustrate the use of RG+quantum for
the detection of suspicious activity on financial data [1]_.

The dataset contains synthethic data generated from a real dataset
of CaixaBank’s express loans [2]_.
Each entry contains, for example, the date and amount of the loan request,
the client identification number and the creation date of the account.
A loan is tagge with either tentative or confirmation of fraud, when a fraudster
has impersonate the client to claim that type of loan and steal client’s funds.

A detailed description of all features is available in [2]_.
"""
# Author: Gregoire Cattan, Filipe Barroso
# License: BSD (3-clause)

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.under_sampling import NearMiss
from pyriemann.preprocessing import Whitening
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import QuanticSVM
import warnings
import pandas as pd
import numpy as np

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

##############################################################################
# Data pre-processing
# ----------------
#
# Download financial data (loan transactions)

url = "https://zenodo.org/record/7418458/files/INFINITECH_synthetic_inmediate_loans.csv"
dataset = pd.read_csv(url, sep=";")

# Transform into binary classification:
# Regroups frauds and suspicions of fraud
dataset.FRAUD[dataset.FRAUD == 2] = 1

# Select a few features for the example
# Note: The choice of these features is not really arbitrary.
# You can use `ydata_profiling` and check these variable are:
#
# 1) Not correlated
# 2) Sufficiently descriminant (based on the number of unique values)
# 3) Are not "empty"
features = dataset[["IP_TERMINAL", "FK_CONTRATO_PPAL_OPE", "SALDO_ANTES_PRESTAMO"]]
target = dataset.FRAUD

# let's display a screenshot of the pre-processed dataset
features.head()
print(f"number of fraudulent loans: {target[target == 1].size}")
print(f"number of genuine loans: {target[target == 0].size}")

# Let's encode our categorical variable (LabelEncoding):
features["IP_TERMINAL"] = features["IP_TERMINAL"].astype("category").cat.codes

# ... and create an 'index' column in the dataset
# Note: this is done only for progamming reason, due to our implementation
# of the `ToEpochs` transformer (see below)
features["index"] = features.index

##############################################################################
# Create the pipeline
# ----------------
#
# Let's create the pipeline as suggested in the patent application

# Let's start by creating the required transformers:


class ToEpochs(TransformerMixin, BaseEstimator):
    def __init__(self, n):
        self.n = n

    def fit(self, X, y):
        return self

    def transform(self, X):
        all_epochs = []
        for x in X:
            id = x[3]
            epoch = features[features.index > id - self.n]
            epoch = epoch[epoch.index <= id]
            epoch.drop(columns=["index"], inplace=True)
            all_epochs.append(epoch)
        all_epochs = np.array(all_epochs)
        return all_epochs


def slim(x, keep_diagonal=True):
    # Vectorize covariance matrices by removing redundant information.
    length = len(x) // 2
    first = range(0, length)
    last = range(len(x) - length, len(x))
    down_cadrans = x[np.ix_(last, last)]
    if keep_diagonal:
        down_cadrans = [down_cadrans[i, j] for i in first for j in first if i <= j]
    else:
        down_cadrans = [down_cadrans[i, j] for i in first for j in first if i < j]
    first_cadrans = np.reshape(x[np.ix_(last, first)], (1, len(x)))
    ret = np.append(first_cadrans, down_cadrans)
    return ret


class SlimVector(TransformerMixin, BaseEstimator):
    def __init__(self, keep_diagonal):
        self.keep_diagonal = keep_diagonal

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([slim(x, self.keep_diagonal) for x in X])


class OptionalWhitening(TransformerMixin, BaseEstimator):
    def __init__(self, process=True, n_components=4):
        self.process = process
        self.n_components = n_components

    def fit(self, X, y):
        return self

    def transform(self, X):
        if not self.process:
            return X
        return Whitening(dim_red={"n_components": 4}).fit_transform(X)


# Finally put together the transformers, and add at the end
# the SVM classifier (classical)
pipe = make_pipeline(
    ToEpochs(n=10),
    XdawnCovariances(nfilter=1),
    OptionalWhitening(process=True, n_components=4),
    SlimVector(keep_diagonal=True),
    SVC(),
)

# Optimize the pipeline.
# Let's save some time and run the optimization with a classical SVM.
gs = GridSearchCV(
    pipe,
    param_grid={
        "toepochs__n": [10, 20],
        "xdawncovariances__nfilter": [1, 2],
        "optionalwhitening__process": [True, False],
        "optionalwhitening__n_components": [2, 4],
        "slimvector__keep_diagonal": [True, False],
    },
    scoring="balanced_accuracy",
)

##############################################################################
# Run evaluation
# ----------------
#
# Balance the data and run the evaluation on a quantum vs classical pipeline.

# We only have about 200 frauds epochs over 30K entries.
# Let's balance the problem using NearMiss.
# Note: at this stage `features` also contains the `index` column.
# So `NearMiss` we choose the closest 200 non-fraud epochs to the 200 fraud-epochs
# based also on this `index` column. This should be improved for real use cases.
X, y = NearMiss().fit_resample(features.to_numpy(), target.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y)

labels, counts = np.unique(y_train, return_counts=True)
print(f"Training set shape: {X_train.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

labels, counts = np.unique(y_test, return_counts=True)
print(f"Testing set shape: {X_test.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

# Let's fit our GridSearchCV, to find the best hyper parameters:
gs.fit(X_train, y_train)

# Print cross-validation results
print(pd.DataFrame.from_dict(gs.cv_results_))

# This is the best score with the classical SVM.
# /!\ Ideally, we should have different datasets for training and validation.
# In a real scenario, we could use some data augmentation techniques, because
# we have only a few samples.
score_svm = gs.best_estimator_.score(X_test, y_test)

# Let's take the same parameters but evaluate the pipeline with a quantum SVM:
gs.best_estimator_.steps[4] = ("quanticsvm", QuanticSVM(quantum=True))
score_qsvm = gs.best_estimator_.fit(X_train, y_train).score(X_test, y_test)

# Print the results
print(f"Classical: {score_svm}; Quantum: {score_qsvm}")

###############################################################################
# References
# ----------
# .. [1] 'SUSPICIOUS ACTIVITY DETECTION USING QUANTUM COMPUTER',
#         Patent application number: 18/380799
# .. [2]  'Synthetic Data of Transactions for Inmediate Loans Fraud'
#         https://zenodo.org/records/7418458
#
#
#
