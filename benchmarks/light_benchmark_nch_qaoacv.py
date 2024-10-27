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
from pyriemann.estimation import XdawnCovariances
from qiskit_algorithms.optimizers import SPSA
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.utils.hyper_params_factory import \
    create_mixer_rotational_X_gates

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

pipelines["NCH_MIN_HULL_QAOACV"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    QuanticNCH(
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        # Provide create_mixer to force QAOA-CV optimization
        create_mixer=create_mixer_rotational_X_gates(0),
        shots=100,
        qaoa_optimizer=SPSA(maxiter=100),
    ),
)

##############################################################################
# Compute score and compare with PR branch
# ------------------------------------------
#
##############################################################################

run(pipelines)
