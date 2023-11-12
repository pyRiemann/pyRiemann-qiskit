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
A loan is tagged with either tentative or confirmation of fraud, when a fraudster
has impersonates the client to claim the loan and steal the client funds.

Once the fraud is caracterized, a complex task is to identify whether or not a collusion
is taking place. One fraudster can for example corrupt a client having already a good
history with the bank. The fraud can also involves a bank agent who is mandated by the client.
The scam perdurates over time, sometime over month or years.
Identifying these participants is essential to prevent similar scam to happen in the future.

In this example, we will use RG to identify whether or not a fraud is a probable collusion.
Because this method work on a small number of components, it is also compatible with Quantum.

"""
# Authors: Gregoire Cattan, Filipe Barroso
# License: BSD (3-clause)

from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import NearMiss, TomekLinks, EditedNearestNeighbours
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
# Some utils function for plotting
# -------------------
#


def plot_ERP(X, title, n=10, ylim=None, add_digest=False):
    epochs = ToEpochs(n=n).transform(X)
    reduced_centered_epochs = NDRobustScaler().fit_transform(epochs)
    fig = plot_waveforms(reduced_centered_epochs, "mean+/-std")
    fig.axes[0].set_title(f"{title} ({len(X)})")
    if ylim is None:
        ylim = []
        for i_channel in range(len(channels)):
            ylim.append(fig.axes[i_channel].get_ylim())
    for i_channel in range(len(channels)):
        fig.axes[i_channel].set_ylim(ylim[i_channel])
    if not add_digest:
        return fig, ylim
    for i_channel in range(len(channels)):
        fig.axes[i_channel].set(ylabel=digest[i_channel])
    return fig, ylim


def merge_2axes(fig1, fig2, file_name1="f1.png", file_name2="f2.png"):
    # Modified from [5]_
    fig1.savefig(file_name1)
    fig2.savefig(file_name2)
    plt.close(fig1)
    plt.close(fig2)

    # inherit figures' dimensions, partially
    w1, w2 = [int(np.ceil(fig.get_figwidth())) for fig in (fig1, fig2)]
    hmax = int(np.ceil(max([fig.get_figheight() for fig in (fig1, fig2)])))

    fig, axes = plt.subplots(1, w1 + w2, figsize=(w1 + w2, hmax))

    # make two axes of desired height proportion
    gs = axes[0].get_gridspec()
    for ax in axes.flat:
        ax.remove()
    ax1 = fig.add_subplot(gs[0, :w1])
    ax2 = fig.add_subplot(gs[0, w1:])

    ax1.imshow(plt.imread(file_name1))
    ax2.imshow(plt.imread(file_name2))

    for ax in (ax1, ax2):
        for side in ("top", "left", "bottom", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

    return fig


def plot_ERPs(X, y, n=10, label0="Fraud", label1="Genuine"):
    fig0, ylim = plot_ERP(X[y == 1], label0, n, add_digest=True)
    fig1, _ = plot_ERP(X[y == 0], label1, n, ylim)
    merge_2axes(fig0, fig1)
    plt.show()


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
channels = [
    "IP_TERMINAL",
    "FK_CONTRATO_PPAL_OPE",
    "POSICION_GLOBAL_ANTES_PRESTAMO",
    "SALDO_ANTES_PRESTAMO",
    "FK_NUMPERSO",
    "FECHA_ALTA_CLIENTE",
    "FK_TIPREL",
]
digest = [
    "IP",
    "Contract",
    "Account balance",
    "Global balance",
    "ID",
    "Date",
    "Ownership",
]
features = dataset[channels]
target = dataset.FRAUD

# let's display a screenshot of the pre-processed dataset
# We only have about 200 frauds epochs over 30K entries.
print(features.head())
print(f"number of fraudulent loans: {target[target == 1].size}")
print(f"number of genuine loans: {target[target == 0].size}")

# Simple treatement for NaN value
features.fillna(method="ffill", inplace=True)

# Convert date value to linux time
features["FECHA_ALTA_CLIENTE"] = pd.to_datetime(features["FECHA_ALTA_CLIENTE"])
features["FECHA_ALTA_CLIENTE"] = features["FECHA_ALTA_CLIENTE"].apply(
    lambda x: 2023 - x.year
)

# features["PK_TSINSERCION"] = pd.to_datetime(features["PK_TSINSERCION"])
# features["PK_TSINSERCION"] = features["PK_TSINSERCION"].apply(lambda x: x.value)

# Let's encode our categorical variables (LabelEncoding):
# features["IP_TERMINAL"] = features["IP_TERMINAL"].astype("category").cat.codes
le = LabelEncoder()
le.fit(features["IP_TERMINAL"].astype("category"))
features["IP_TERMINAL"] = le.transform(features["IP_TERMINAL"].astype("category"))

# ... and create an 'index' column in the dataset
# Note: this is done only for progamming reasons, due to our implementation
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
            epoch = features[features.index > index - self.n - 1]
            epoch = epoch[epoch.index < index]
            epoch.drop(columns=["index"], inplace=True)
            all_epochs.append(np.transpose(epoch))
        all_epochs = np.array(all_epochs)
        return all_epochs


# Apply one scaler by channel:
# See Stackoverflow link for more details [4]_
class NDRobustScaler(TransformerMixin):
    def __init__(self):
        self._scalers = []

    def fit(self, X, y=None, **kwargs):
        _, n_channels, _ = X.shape
        self._scalers = []
        for i in range(n_channels):
            scaler = RobustScaler()
            scaler.fit(X[:, i, :])
            self._scalers.append(scaler)
        return self

    def transform(self, X, **kwargs):
        n_channels = len(self._scalers)
        for i in range(n_channels):
            X[:, i, :] = self._scalers[i].transform(X[:, i, :])
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


# Create a RandomForest for baseline comparison of direct classification:
rf = RandomForestClassifier()

# Classical pipeline: puts together transformers,
# then adds at the end a classical SVM
pipe = make_pipeline(
    ToEpochs(n=10),
    NDRobustScaler(),
    XdawnCovariances(nfilter=1),
    OptionalWhitening(process=True, n_components=4),
    SlimVector(keep_diagonal=True),
    SVC(probability=True),
)

# Optimize the pipeline:
# let's save some time and run the optimization with the classical SVM
gs = HalvingGridSearchCV(
    pipe,
    param_grid={
        "toepochs__n": [10, 20, 30],
        "xdawncovariances__nfilter": [1, 2, 3],
        "optionalwhitening__process": [True, False],
        "optionalwhitening__n_components": [2, 4, 6],
        "slimvector__keep_diagonal": [True, False],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["auto", "scale", 1, 10],
    },
    scoring="balanced_accuracy",
    cv=4,
    min_resources="smallest",
    verbose=1,
)


##############################################################################
# Balance dataset
# ---------------
#
# Balance the data and display the "ERP" [3]_.

# Let's balance the problem using NearMiss.
# There is no special need to do this with SVM, as we could also play with the `class_weight`
# parameter. However, reducing the data is practical for Ci/Cd.
# So `NearMiss` we choose the closest non-fraud epochs to the fraud-epochs.
# Here we will keep a ratio of 2 non-fraud epochs for 1 fraud epochs.
# Note: at this stage `features` also contains the `index` column.
X, y = NearMiss(sampling_strategy=0.5).fit_resample(
    features.to_numpy(), target.to_numpy()
)
# X, y = EditedNearestNeighbours().fit_resample(features.to_numpy(), target.to_numpy())
# X = features.to_numpy()
# y = target.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

labels, counts = np.unique(y_train, return_counts=True)
print(f"Training set shape: {X_train.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

labels, counts = np.unique(y_test, return_counts=True)
print(f"Testing set shape: {X_test.shape}, genuine: {counts[0]}, frauds: {counts[1]}")

##############################################################################
# Run evaluation
# --------------
#
# (Supervised classification)
#
# Run the evaluation on a classical vs quantum pipeline.

# Let's fit our GridSearchCV, to find the best hyper parameters
gs.fit(X_train, y_train)

# Print best parameters
print("Best parameters are:")
print(gs.best_params_)
best_C = gs.best_params_["svc__C"]
best_gamma = gs.best_params_["svc__gamma"]
best_n = gs.best_params_["toepochs__n"]

# This is the best score with the classical SVM.
# (with this train/test split at least)
train_pred_svm = gs.best_estimator_.predict(X_train)
train_score_svm = balanced_accuracy_score(y_train, train_pred_svm)
pred_svm = gs.best_estimator_.predict(X_test)
score_svm = balanced_accuracy_score(y_test, pred_svm)

# Quantum pipeline:
# let's take the same parameters but evaluate the pipeline with a quantum SVM.
# Note: From experience, quantum SVM tends to overfit quickly.
# So it is debatable if we want to keep the same penalties for the quantum SVM as for the classical one.
gs.best_estimator_.steps[-1] = (
    "quanticsvm",
    QuanticSVM(quantum=True, C=best_C, gamma=best_gamma),
)
train_pred_qsvm = gs.best_estimator_.fit(X_train, y_train).predict(X_train)
train_score_qsvm = balanced_accuracy_score(y_train, train_pred_qsvm)
pred_qsvm = gs.best_estimator_.predict(X_test)
score_qsvm = balanced_accuracy_score(y_test, pred_qsvm)

# Create a point of comparison with the RandomForest
train_pred_rf = rf.fit(X_train, y_train).predict(X_train)
train_score_rf = balanced_accuracy_score(y_train, train_pred_rf)
pred_rf = rf.predict(X_test)
score_rf = balanced_accuracy_score(y_test, pred_rf)

# Print the results of direct classification of fraud records
# Note:
# SVM/QSVM pipeline use the loans preceding the actual fraud, without the fraud itself
# RandomForest uses only the fraud record itself
#
print("----Training score:----")
print(
    f"Classical SVM: {train_score_svm}\
    \nQuantum SVM: {train_score_qsvm}\
    \nClassical RandomForest: {train_score_rf}"
)
print("----Testing score:----")
print(
    f"Classical SVM: {score_svm}\
    \nQuantum SVM: {score_qsvm}\
    \nClassical RandomForest: {score_rf}"
)


##############################################################################
#
# Unsupervised classification (collusion vs no-collusion)
#
# We will now predict whether or not the fraud was a collusion or not.
# This is a two-stage process:
# 1) We have the no-aware ERP method (namely RandomForest)
#    to predict whether or not the transaction is a fraud
# 2) If the fraud is caracterized, we use the QSVC pipeline to
#    predict whether or not it is a collusion
#
class ERP_CollusionClassifier(ClassifierMixin):
    def __init__(self, row_clf, erp_clf, threshold=0.7):
        self.row_clf = row_clf
        self.erp_clf = erp_clf
        self.threshold = threshold

    def fit(self, X, y):
        # Do not apply: Classifiers are already fitted
        return self

    def predict(self, X):
        y_pred = self.row_clf.predict(X)
        collusion_prob = self.erp_clf.predict_proba(X)
        y_pred[y_pred == 1] = collusion_prob[y_pred == 1, 1].transpose()
        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0
        return y_pred


# plot the temporal response of collusion vs no-collusion.
y_pred = ERP_CollusionClassifier(gs.best_estimator_, rf).predict(X)
plot_ERPs(X, y_pred, best_n, "Collusion", "No-fraud & Fraud without collusion")

# Let's predict the type of fraud using our two-stage:
# The y_pred here contains 1 if the fraud is a possible collusion or 0 else, i.e:
# not a fraud or not a collusion fraud
y_pred = ERP_CollusionClassifier(gs.best_estimator_, rf).predict(X_test)

# We will get the epochs associated with these frauds
high_warning_loan = np.concatenate(ToEpochs(n=best_n).transform(X_test[y_pred == 1]))

# and from there the IPs of incriminated terminals and and the IDs of the suspicious customers
# for further investigation:
high_warning_ip = le.inverse_transform(high_warning_loan[0, :].astype(int))
high_warning_id = high_warning_loan[3, :].astype(str)
print("IP involved in probable collusion: ", high_warning_ip)
print("ID involved in probable collusion: ", high_warning_id)


###############################################################################
# References
# ----------
# .. [1] 'SUSPICIOUS ACTIVITY DETECTION USING QUANTUM COMPUTER',
#         Patent application number: 18/380799
# .. [2] 'Synthetic Data of Transactions for Inmediate Loans Fraud'
#         https://zenodo.org/records/7418458
# .. [3] https://pyriemann.readthedocs.io/en/latest/auto_examples/ERP/plot_ERP.html
# .. [4] https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
# .. [5] https://stackoverflow.com/questions/16748577/matplotlib-combine-different-figures-and-put-them-in-a-single-subplot-sharing-a
#
