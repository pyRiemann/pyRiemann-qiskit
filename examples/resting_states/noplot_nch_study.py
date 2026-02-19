"""
====================================================================
Classification of P300 datasets from MOABB using NCH
====================================================================

Comparison of NCH with different optimization methods,
in a "hard" dataset (classical methods don't provide results).

"""
# Author: Gregoire Cattan, Quentin Barthelemy
# Modified from noplot_classify_P300_nch.py
# License: BSD (3-clause)

import random

import numpy as np
from pyriemann_qiskit.classification.qaoa_batch_classifier import QAOABatchClassifier
import qiskit_algorithms
import seaborn as sns
from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import Cattan2019_PHMD, Rodrigues2017, Hinss2021
from moabb.evaluations import CrossSubjectEvaluation, WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from pyriemann.spatialfilters import CSP
from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils.hyper_params_factory import create_mixer_identity, create_mixer_with_circular_entanglement, create_mixer_rotational_XY_gates, create_mixer_rotational_X_gates
from pyriemann_qiskit.utils.anderson_optimizer import AndersonAccelerationOptimizer

# import warnings


print(__doc__)

##############################################################################
# getting rid of the warnings about the future
# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)

# warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Set global seed for better reproducibility
#seed = 475751
import time
seed = round(time.time())
print(seed)

random.seed(seed)
np.random.seed(seed)
qiskit_algorithms.utils.algorithm_globals.random_seed = seed

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.


#events=["open", "closed"]
#

# The paradigm is adapted to the P300 paradigm.

#Rodrigues2017
#events = dict(closed=1, open=2)
#paradigm = RestingStateToP300Adapter(events=events)

# Hinss2021 CrossSubjectEvaluation CrossSessionEvaluation->in article
#events = dict(easy=2, medium=3)
#paradigm = RestingStateToP300Adapter(events=events, tmin=0, tmax=0.5)

# Cattan2019_PHMD()
#events = ["on", "off"]
events = dict(on=0, off=1)
paradigm = RestingStateToP300Adapter(events=events)

datasets = [
    Cattan2019_PHMD(), # CrossSubjectEvaluation
    #Rodrigues2017(), # WithinSessionEvaluation CrossSubjectEvaluation
   # Hinss2021(), CrossSubjectEvaluation CrossSessionEvaluation
]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}
pipelines2 = {}

n_hulls_per_class = 3
n_samples_per_hull = 6

sf = make_pipeline(
    Covariances(estimator="lwf"),
    
)

##############################################################################
# NCH

optimizer = AndersonAccelerationOptimizer(
    maxiter=15,      # Fewer iterations with Anderson
    m=5,             # History depth
    alpha=1.0,       # Undamped
    beta=0.01        # Step size for smooth functions
)

pipelines2["NCH+MIN_HULL_QAOACV(Ulvi)"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        #n_hulls_per_class=n_hulls_per_class,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        create_mixer= create_mixer_with_circular_entanglement(0),#create_mixer_identity(),
        shots=100,
        qaoa_optimizer=L_BFGS_B(maxiter=100, maxfun=200),
        #qaoa_optimizer=optimizer,
        n_reps=2,
        qaoacv_implementation="ulvi"
    ),
)


pipelines2["NCH+MIN_HULL_NAIVEQAOA"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
       # qaoa_optimizer=SLSQP(),
        quantum=True,
        n_reps=2
    ),
)

##############################################################################
# SOTA classical methods for comparison
pipelines["MDM"] = make_pipeline(
    sf,
    #CSP(nfilter=4, log=False),
    MDM(),
)

pipelines["TS+LDA"] = make_pipeline(
    sf,
    CSP(nfilter=4, log=False),
    TangentSpace(metric="riemann"),
    #LDA(),
    QAOABatchClassifier(optimizer=AndersonAccelerationOptimizer())
)

print("Total pipelines to evaluate: ", len(pipelines))

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    suffix="examples",
    overwrite=overwrite,
    n_splits=3,
    random_state=seed,
    #shuffle=True,
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
    "NCH+MIN_HULL_QAOACV(Ulvi)",
    "NCH+MIN_HULL_NAIVEQAOA",
    "TS+LDA",
    "MDM",
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
#ax.set_ylim(0.35, 0.7)
plt.xticks(rotation=45)
plt.show()

