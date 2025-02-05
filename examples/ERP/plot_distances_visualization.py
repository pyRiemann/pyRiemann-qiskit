"""
====================================================================
Visualize distances with MDM and NCH
====================================================================

Demonstrates how to use the visualization module,
to plot the distances between the classes.

"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import warnings

from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.paradigms import P300
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from pyriemann.preprocessing import Whitening
from pyriemann.utils.viz import plot_bihist, plot_biscatter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.visualization.manifold import plot_manifold

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)

# warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Create Pipelines
# ----------------
#
# Create a MDM and NCH pipeline, as well as an estimator for 2x2 cov matrices.


paradigm = P300(resample=128)

ds = bi2012()

# Change this to use NCH instead of MDM estimator for the distances
USE_MDM = True

mdm = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    MDM(metric="logeuclid"),
)

nch = make_pipeline(
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


estimator = mdm if USE_MDM else nch

cov2x2 = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        estimator="scm",
        xdawn_estimator="lwf",
    ),
    Whitening(dim_red={"n_components": 2}),
)

##############################################################################
# Data
# ----------------
#
# Retrieve data from bi2012


X, y, _ = paradigm.get_data(ds, subjects=[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

##############################################################################
# Plot manifold
# ----------------
#
# Plot cov matrices in 3d cartesian space.
# (they form a cone)

points = cov2x2.fit(X_train, y_train).transform(X_test)
plot_manifold(points, y_test, False)


##############################################################################
# Plot distances
# ----------------
#
# Plot distances between the classes with the MDM or NCH estimator.

dists = estimator.fit(X_train, y_train).transform(X_test)

plot_biscatter(dists, y_test)
plot_bihist(dists, y_test)

plt.show()
