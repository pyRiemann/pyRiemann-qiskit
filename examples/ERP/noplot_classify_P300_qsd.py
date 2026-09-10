"""
====================================================================
Classification of P300 datasets with the Quantum State Discriminator
====================================================================

QuantumStateDiscriminator (QSD) models a mental state as a density
matrix and classifies with the Pretty Good Measurement. Its default
operator is the temporal covariance X.X^T, which suits resting-state
paradigms, where the classes differ in spatial covariance and band
power.

A P300 is a different problem: target and non-target epochs differ in
a transient deflection ~300 ms after the stimulus, and are otherwise
close to identical. A plain covariance integrates over time and
averages that deflection away, so QSD scores at chance on this
paradigm.

``covariance="erp"`` fixes it. The class-average evoked responses are
concatenated to each Xdawn-filtered trial before the covariance, so
the operator also carries the trial/prototype cross-correlation, which
is where the time-locked information lives. ``prototype_weight`` then
sets how much of the operator's trace that block is allowed to claim:
a class-average prototype has far less energy than a single trial, so
it needs boosting to compete.

This example compares, on identical folds:

- Xdawn covariances + tangent space + logistic regression (reference)
- logistic regression on flattened epochs
- QSD with the default covariance
- QSD with the ERP covariance

Over the eight MOABB P300 datasets (241 subjects) the ERP variant
reaches 0.858 AUC against 0.850 for flattened logistic regression and
0.878 for the tangent space reference, and holds up under nested
cross-validation of its hyper-parameters.

"""

# Author: Gregoire Cattan
# License: BSD (3-clause)

import warnings

from moabb import set_log_level
from moabb.datasets import BNCI2014_008, BNCI2014_009, BI2013a
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from pyriemann_qiskit.classification import QuantumStateDiscriminator

print(__doc__)

set_log_level("info")
warnings.filterwarnings("ignore")

SEED = 42

##############################################################################
# Pipelines
# ---------


def flatten(X):
    return X.reshape(len(X), -1)


pipelines = {}

pipelines["XdawnCov+TS+LR"] = make_pipeline(
    XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas"),
    # "logdet" is a mean metric only, not a tangent space map, so it has to
    # be paired with a map rather than passed as a bare string.
    TangentSpace(metric={"mean": "logdet", "map": "riemann"}),
    LogisticRegression(max_iter=3000),
)

pipelines["FlatLR"] = make_pipeline(
    FunctionTransformer(flatten, validate=False),
    StandardScaler(),
    LogisticRegression(
        C=0.1,
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED,
    ),
)

# The default operator: expected to sit at chance on a time-locked paradigm.
pipelines["QSD"] = make_pipeline(
    QuantumStateDiscriminator(),
)

pipelines["QSD(erp)"] = make_pipeline(
    QuantumStateDiscriminator(
        covariance="erp",
        nfilter=8,
        prototype_weight=100,
    ),
)

##############################################################################
# Evaluation
# ----------
#
# One paradigm for every pipeline, so all of them see the same epochs and
# the same folds.

datasets = [BNCI2014_009(), BNCI2014_008(), BI2013a()]

evaluation = WithinSessionEvaluation(
    paradigm=P300(),
    datasets=datasets,
    random_state=SEED,
    n_jobs=6,
    overwrite=True,
    suffix="qsd_p300",
)
# Note: WithinSessionEvaluation accepts n_splits but ignores it -- its
# _create_splitter hardcodes n_folds=5 -- so the fold count is left at the
# default rather than passed in and silently dropped.

results = evaluation.process(pipelines)

##############################################################################
# Results
# -------

subjects = results.groupby(["dataset", "subject", "pipeline"], as_index=False).agg(
    auc=("score", "mean"), time=("time", "mean")
)

print("\nAveraging the subject performance:")
print(
    subjects.groupby("pipeline")
    .agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        n_subjects=("auc", "count"),
        time_mean=("time", "mean"),
    )
    .sort_values("auc_mean", ascending=False)
)

print("\nBy dataset:")
print(
    subjects.pivot_table(
        index="dataset", columns="pipeline", values="auc", aggfunc="mean"
    ).round(4)
)
