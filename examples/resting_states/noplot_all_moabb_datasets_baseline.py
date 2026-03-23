"""
====================================================================
Baseline classification of resting-state datasets from MOABB using CSP and MDM
====================================================================

Within-session evaluation of CSP+LDA and MDM pipelines across three
resting-state datasets: Hinss2021, Rodrigues2017, and Cattan2019_PHMD.
No transfer learning is applied.

"""
# Author: Gregoire Cattan
# Modified from noplot_all_moabb_datasets.py
# License: BSD (3-clause)

import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)
from moabb.analysis.plotting import summary_plot
from moabb.datasets import Cattan2019_PHMD, Hinss2021, Rodrigues2017
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

print(__doc__)

set_log_level("info")

##############################################################################
# Set global seed for better reproducibility

seed = round(time.time())
print(seed)

random.seed(seed)
np.random.seed(seed)

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

sf = make_pipeline(
    Covariances(estimator="lwf"),
)

pipelines = {}

pipelines["CSP+MDM"] = make_pipeline(
    sf,
    CSP(nfilter=8, log=False),
    MDM(),
)

print("Total pipelines to evaluate: ", len(pipelines))

overwrite = True  # set to True if we want to overwrite cached results

##############################################################################
# Evaluations
# -----------
#
# One WithinSession evaluation per dataset. Each dataset requires its own
# paradigm due to different event codes.

# --- Cattan2019_PHMD: WithinSession ---
paradigm_cattan = RestingStateToP300Adapter(events=dict(on=0, off=1))
evaluation_cattan = WithinSessionEvaluation(
    paradigm=paradigm_cattan,
    datasets=[Cattan2019_PHMD()],
    suffix="baseline_cattan",
    overwrite=overwrite,
    n_splits=3,
)
results_cattan = evaluation_cattan.process(pipelines)
print("Cattan pipelines computed:", results_cattan["pipeline"].unique().tolist())
print("Cattan results:")
print(results_cattan.groupby("pipeline")[["score", "time"]].mean())

# --- Rodrigues2017: WithinSession ---
paradigm_rodrigues = RestingStateToP300Adapter(events=dict(closed=2, open=1))
evaluation_rodrigues = WithinSessionEvaluation(
    paradigm=paradigm_rodrigues,
    datasets=[Rodrigues2017()],
    suffix="baseline_rodrigues",
    overwrite=overwrite,
    n_splits=3,
)
results_rodrigues = evaluation_rodrigues.process(pipelines)
print("Rodrigues results:")
print(results_rodrigues.groupby("pipeline")[["score", "time"]].mean())

# --- Hinss2021: WithinSession ---
paradigm_hinss = RestingStateToP300Adapter(
    events=dict(easy=2, medium=3), tmin=0, tmax=0.5
)
evaluation_hinss = WithinSessionEvaluation(
    paradigm=paradigm_hinss,
    datasets=[Hinss2021()],
    suffix="baseline_hinss",
    overwrite=overwrite,
    n_splits=3,
)
results_hinss = evaluation_hinss.process(pipelines)
print("Hinss results:")
print(results_hinss.groupby("pipeline")[["score", "time"]].mean())

##############################################################################
# Aggregate Results
# -----------------

results = pd.concat(
    [results_cattan, results_rodrigues, results_hinss],
    ignore_index=True,
)

print(results)
print("Averaging the session performance:")
print(results.groupby("pipeline")[["score", "time"]].mean())

##############################################################################
# Plot Results: Raw Scores per Dataset
# -------------------------------------
#
# Strip + point plot of per-subject scores, with dataset on x-axis.

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="dataset",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(
    data=results,
    y="score",
    x="dataset",
    ax=ax,
    palette="Set1",
)

ax.set_ylabel("ROC AUC")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
