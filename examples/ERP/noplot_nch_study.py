"""
====================================================================
Classification of P300 datasets from MOABB using NCH
====================================================================

Comparison of NCH with different optimization methods,
in a "hard" dataset (classical methods don't provide results)

"""
# Author: Gregoire Cattan
# Modified from noplot_classify_P300_nch.py
# License: BSD (3-clause)

import warnings
import numpy as np
import random
import qiskit_algorithms

from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import bi2013a, bi2012, Cattan2019_VR, Cattan2019_PHMD
from moabb.datasets.compound_dataset import Cattan2019_VR_Il
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import P300, RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, Covariances, Shrinkage, ERPCovariances
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann_qiskit.pipelines import QuantumMDMWithRiemannianPipeline
from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils.hyper_params_factory import create_mixer_rotational_X_gates, create_mixer_rotational_XY_gates
from pyriemann.spatialfilters import CSP

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Set global seed for better reproducibility
seed = 475751

random.seed(seed)
np.random.seed(seed)
qiskit_algorithms.utils.algorithm_globals.random_seed

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

events = ["on", "off"]
paradigm = RestingStateToP300Adapter(events=events)

datasets = [Cattan2019_PHMD()]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

n_hulls_per_class = 3
n_samples_per_hull = 6

sf = make_pipeline(
    Covariances(estimator="lwf"),
)

##############################################################################
# NCH without quantum optimization
pipelines["NCH+RANDOM_HULL"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=n_hulls_per_class,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="random",
        quantum=False,
    ),
)

pipelines["NCH+MIN_HULL"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=False,
    ),
)


##############################################################################
# NCH with quantum optimization
pipelines["NCH+RANDOM_HULL_QAOACV"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=n_hulls_per_class,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="random",
        quantum=True,
        create_mixer=create_mixer_rotational_X_gates(0),
        shots=100,
        qaoa_optimizer=SPSA(maxiter=100, blocking=False),
        n_reps=2
    ),
)

pipelines["NCH+RANDOM_HULL_NAIVEQAOA"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=n_hulls_per_class,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="random",
        quantum=True,
    ),
)

pipelines["NCH+MIN_HULL_QAOACV"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        create_mixer=create_mixer_rotational_X_gates(0),
        shots=100,
        qaoa_optimizer=SPSA(maxiter=100, blocking=False),
        n_reps=2
    ),
)

pipelines["NCH+MIN_HULL_NAIVEQAOA"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=True,
    ),
)

##############################################################################
# SOTA classical methods for comparison
pipelines["MDM"] = make_pipeline(
    sf,
    MDM(),
)

pipelines["TS+LDA"] = make_pipeline(
      sf,
      TangentSpace(metric="riemann"),
      LDA(),
  )

print("Total pipelines to evaluate: ", len(pipelines))

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite,
    n_splits=3,
    random_state=seed,
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

order = [
    'NCH+RANDOM_HULL',
    'NCH+RANDOM_HULL_NAIVEQAOA',
    'NCH+RANDOM_HULL_QAOACV',
    'NCH+MIN_HULL',
    'NCH+MIN_HULL_NAIVEQAOA',
    'NCH+MIN_HULL_QAOACV',
    'TS+LDA',
    'MDM'
]

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
    order=order,
    hue_order=order,
)
sns.pointplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    palette="Set1",
    order=order,
    hue_order=order,
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 0.75)
plt.xticks(rotation=45)
plt.show()
