"""
====================================================================
Light Benchmark
====================================================================

This benchmark is a non-regression performance test, intended
to run on Ci with each PRs.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_P300_bi.py of pyRiemann
# License: BSD (3-clause)

import warnings

from lb_base import run
from moabb import set_log_level
from pyriemann.estimation import Shrinkage, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
    QuantumMDMWithRiemannianPipeline,
)
from pyriemann_qiskit.utils import distance, mean  # noqa

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
#
##############################################################################

pipelines = {}

pipelines["RG_QSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100,
    nfilter=2,
    dim_red=PCA(n_components=5),
    params={"seed": 42, "use_fidelity_state_vector_kernel": True},
)

pipelines["RG_VQC"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100, spsa_trials=1, two_local_reps=2, params={"seed": 42}
)

pipelines["QMDM_mean"] = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "qeuclid", "distance": "euclid"},
    quantum=True,
    regularization=Shrinkage(shrinkage=0.9),
    shots=1024,
    seed=696288,
)

pipelines["QMDM_dist"] = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "logeuclid", "distance": "qlogeuclid_hull"},
    quantum=True,
    seed=42,
    shots=100,
)

pipelines["RG_LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    PCA(n_components=5),
    LDA(solver="lsqr", shrinkage="auto"),
)

##############################################################################
# Compute score and compare with PR branch
# ------------------------------------------
#
##############################################################################

run(pipelines)
