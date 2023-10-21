"""
====================================================================
Suspicious financial activity detection using quantum computer
====================================================================

In this example, we will illustrate the use of Riemannian geometry and quantum
computing for the detection of suspicious activity on financial data [1]_.

The dataset contains synthethic data generated from a real dataset
of CaixaBank’s express loans [2]_.
Each entry contains, for example, the date and amount of the loan request,
the client identification number and the creation date of the account.
A loan is tagge with either tentative or confirmation of fraud, when a fraudster
has impersonate the client to claim that type of loan and steal client’s funds.

A detailed description of all features is available in [2]_.
"""
# Authors: Gregoire Cattan, Filipe Barroso
# License: BSD (3-clause)

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import NearMiss
from pyriemann.preprocessing import Whitening
from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.viz import plot_waveforms
from pyriemann_qiskit.classification import QuanticSVM
from matplotlib import pyplot as plt
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
# -------------------
#
# Pre-process financial data (loan transactions)

# Download data
url = "https://zenodo.org/record/7418458/files/INFINITECH_synthetic_inmediate_loans.csv"
dataset = pd.read_csv(url, sep=";")

# Transform into binary classification, regroup frauds and suspicions of fraud
dataset.FRAUD[dataset.FRAUD == 2] = 1

# Select a few features for the example
# Note: The choice of these features is not really arbitrary.
# You can use `ydata_profiling` and check these variable are:
#
# 1) Not correlated
# 2) Sufficiently descriminant (based on the number of unique values)
# 3) Are not "empty"
channels = ["IP_TERMINAL", "FK_CONTRATO_PPAL_OPE", "SALDO_ANTES_PRESTAMO", "FK_NUMPERSO", "FECHA_ALTA_CLIENTE", "FK_TIPREL"]
digest   = ["IP", "Contract code", "Balance", "ID", "Seniority", "Ownership"]
features = dataset[channels]
target = dataset.FRAUD

# let's display a screenshot of the pre-processed dataset
# We only have about 200 frauds epochs over 30K entries.

print(features.head())
print(f"number of fraudulent loans: {target[target == 1].size}")
print(f"number of genuine loans: {target[target == 0].size}")

# Simple treatement for NaN value
features.fillna(method='ffill', inplace=True)

# Convert date value to linux time
features['FECHA_ALTA_CLIENTE'] = pd.to_datetime(features['FECHA_ALTA_CLIENTE'])
features['FECHA_ALTA_CLIENTE'] = features['FECHA_ALTA_CLIENTE'].apply(lambda x: x.value)

# Let's encode our categorical variable (LabelEncoding):
features["IP_TERMINAL"] = features["IP_TERMINAL"].astype("category").cat.codes

# ... and create an 'index' column in the dataset
# Note: this is done only for progamming reason, due to our implementation
# of the `ToEpochs` transformer (see below)
features["index"] = features.index


##############################################################################
# Pipeline for binary classification
# ----------------------------------
#
# Let's create the pipeline as suggested in the patent application [1]_.

# Let's start by creating the required transformers:


class ToEpochs(TransformerMixin, BaseEstimator):
    def __init__(self, n):
        self.n = n

    def fit(self, X, y):
        return self

    def transform(self, X):
        all_epochs = []
        for x in X:
            index = x[-1]
            epoch = features[features.index > index - self.n]
            epoch = epoch[epoch.index <= index]
            epoch.drop(columns=["index"], inplace=True)
            all_epochs.append(np.transpose(epoch))
        all_epochs = np.array(all_epochs)
        return all_epochs


# Stackoverflow implementation [4]_
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, y=None, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, y, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


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


# Classical pipeline: put together the transformers, and add at the end
# the classical SVM
pipe = make_pipeline(
    ToEpochs(n=10),
    NDStandardScaler(),
    XdawnCovariances(nfilter=1),
    OptionalWhitening(process=True, n_components=4),
    SlimVector(keep_diagonal=True),
    SVC(),
)

# Optimize the pipeline:
# let's save some time and run the optimization with the classical SVM
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
# Balance dataset
# ---------------
#
# Balance the data and display the "ERP" [3]_.

# Let's balance the problem using NearMiss.
# Note: at this stage `features` also contains the `index` column.
# So `NearMiss` we choose the closest 200 non-fraud epochs to the 200 fraud-epochs.
X, y = NearMiss().fit_resample(features.to_numpy(), target.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

labels, counts = np.unique(y_train, return_counts=True)
print(f"Training set shape: {X_train.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

labels, counts = np.unique(y_test, return_counts=True)
print(f"Testing set shape: {X_test.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

# Before fitting the GridSearchCV, let's display the "ERP"
epochs = ToEpochs(n=10).transform(X_train)
reduced_centered_epochs = NDStandardScaler().fit_transform(epochs)

fig = plot_waveforms(reduced_centered_epochs, "hist")
for i_channel in range(len(channels)):
    fig.axes[i_channel].set(ylabel=digest[i_channel])
plt.show()


##############################################################################
# Run evaluation
# --------------
#
# Run the evaluation on a classical vs quantum pipeline.

# Let's fit our GridSearchCV, to find the best hyper parameters
gs.fit(X_train, y_train)

# Print best parameters
print("Best parameters are:")
print(gs.best_params_)

# This is the best score with the classical SVM.
# (with this train/test split at least)
train_score_svm = gs.best_estimator_.score(X_train, y_train)
score_svm = gs.best_estimator_.score(X_test, y_test)

# Quantum pipeline:
# let's take the same parameters but evaluate the pipeline with a quantum SVM:
gs.best_estimator_.steps[-1] = ("quanticsvm", QuanticSVM(quantum=True))
train_score_qsvm = gs.best_estimator_.fit(X_train, y_train).score(X_train, y_train)
score_qsvm = gs.best_estimator_.score(X_test, y_test)

# Additionally, run a RandomForest for baseline comparison:
rf = RandomForestClassifier()
train_score_rf = rf.fit(X_train, y_train).score(X_train, y_train)
score_rf = rf.score(X_test, y_test)

# Print the results
print(
    f"(Train) Classical: {train_score_svm} \nQuantum: {train_score_qsvm} \nRF: {train_score_rf}"
)
print(f"(Test) Classical: {score_svm} \nQuantum: {score_qsvm} \nRF: {score_rf}")


###############################################################################
# References
# ----------
# .. [1] 'SUSPICIOUS ACTIVITY DETECTION USING QUANTUM COMPUTER',
#         Patent application number: 18/380799
# .. [2] 'Synthetic Data of Transactions for Inmediate Loans Fraud'
#         https://zenodo.org/records/7418458
# .. [3] https://pyriemann.readthedocs.io/en/latest/auto_examples/ERP/plot_ERP.html
# .. [4] https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
#
#
