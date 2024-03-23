"""
====================================================================
Classification of P300 datasets from MOABB using NCH
====================================================================

Demonstrates classification with QunatumNCH.
Evaluation is done using MOABB.

If parameter "shots" is None then a classical SVM is used similar to the one
in scikit learn.
If "shots" is not None and IBM Qunatum token is provided with "q_account_token"
then a real Quantum computer will be used.
You also need to adjust the "n_components" in the PCA procedure to the number
of qubits supported by the real quantum computer you are going to use.
A list of real quantum  computers is available in your IBM quantum account.

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

import warnings

from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import bi2013a, bi2012, Cattan2019_PHMD
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300, RestingStateToP300Adapter 
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
import seaborn as sns
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.visualization.distances import plot_bihist, plot_cone, plot_scatter

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


paradigm = P300(resample=128)

ds = bi2012()

mdm = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    MDM(
    ),
)

nch = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    QuanticNCH(
        n_hulls_per_class=3,
        n_samples_per_hull=15,
        n_jobs=12,
        subsampling="min",
        quantum=False,
    ),
)

from pyriemann.preprocessing import Whitening
xd = make_pipeline(XdawnCovariances(
        nfilter=3,
        estimator="scm",
        xdawn_estimator="lwf",
    ), Whitening(dim_red={"n_components": 2})
    
                  )

X, y, _ = paradigm.get_data(ds, subjects=[1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# dists = mdm.fit(X_train, y_train).transform(X_test)
points = xd.fit(X_train, y_train).transform(X_test)

plot_cone(points, y_test)
plt.show()