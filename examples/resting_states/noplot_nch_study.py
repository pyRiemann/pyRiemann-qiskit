"""
====================================================================
Classification of resting-state datasets from MOABB using NCH
====================================================================

Comparison of classical pipelines (MDM, TS+LDA) across three
resting-state datasets: Hinss2021 (CrossSubject & CrossSession),
Rodrigues2017 (CrossSubject), and Cattan2019_PHMD (CrossSubject).
Results are aggregated and analyzed with MOABB statistical tools.

"""
# Author: Gregoire Cattan, Quentin Barthelemy
# Modified from noplot_classify_P300_nch.py
# License: BSD (3-clause)

import itertools
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
from moabb.analysis.plotting import meta_analysis_plot, summary_plot
from moabb.datasets import Cattan2019_PHMD, Hinss2021, Rodrigues2017
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import MDWM, TLCenter, TLClassifier, TLRotate, TLScale
from qiskit_algorithms.optimizers import L_BFGS_B
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils.hyper_params_factory import (
    create_mixer_with_circular_entanglement,
)
from pyriemann_qiskit.utils.transfer import Adapter, TLCrossSubjectEvaluation

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
   # Whitening(metric="riemann"),
)

n_samples_per_hull = 6

pipelines = {}

pipelines["MDM"] = make_pipeline(
    sf,
    MDM(),
)

pipelines["TS+LDA"] = make_pipeline(
    sf,
    TangentSpace(metric="riemann"),
    LDA(),
)

# pipelines["NCH+MIN_HULL_QAOACV(Ulvi)"] = make_pipeline(
#     sf,
#     QuanticNCH(
#         seed=seed,
#         n_samples_per_hull=n_samples_per_hull,
#         n_jobs=12,
#         subsampling="min",
#         quantum=True,
#         create_mixer=create_mixer_with_circular_entanglement(0),
#         shots=100,
#         qaoa_optimizer=L_BFGS_B(maxiter=100, maxfun=200),
#         n_reps=2,
#         qaoacv_implementation="ulvi",
#     ),
# )

# pipelines["NCH+MIN_HULL_NAIVEQAOA"] = make_pipeline(
#     sf,
#     QuanticNCH(
#         seed=seed,
#         n_samples_per_hull=n_samples_per_hull,
#         n_jobs=12,
#         subsampling="min",
#         quantum=True,
#         n_reps=2,
#     ),
# )

cov = Covariances(estimator="lwf")

pipelines["MDM+TL"] = Adapter(
    preprocessing=cov,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(target_domain=None, estimator=MDM(), domain_weight=None),
    ),
)

pipelines["TS+LDA+TL"] = Adapter(
    preprocessing=cov,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(
            target_domain=None,
            estimator=make_pipeline(TangentSpace(metric="riemann"), LDA()),
            domain_weight=None,
        ),
    ),
)

pipelines["MDWM(0.5)"] = Adapter(
    preprocessing=cov,
    estimator=MDWM(domain_tradeoff=0.5, target_domain=None, metric="riemann"),
)

print("Total pipelines to evaluate: ", len(pipelines))

overwrite = True  # set to True if we want to overwrite cached results

##############################################################################
# Evaluations
# -----------
#
# One CrossSubject evaluation per dataset. Each dataset requires its own
# paradigm due to different event codes.

# --- Cattan2019_PHMD: CrossSubject ---
paradigm_cattan = RestingStateToP300Adapter(events=dict(on=0, off=1))
evaluation_cattan = TLCrossSubjectEvaluation(
    paradigm=paradigm_cattan,
    datasets=[Cattan2019_PHMD()],
    suffix="nch_study_cattan",
    overwrite=overwrite,
    n_splits=3,
    random_state=seed,
)
results_cattan = evaluation_cattan.process(pipelines)
print("Cattan pipelines computed:", results_cattan["pipeline"].unique().tolist())

# --- Rodrigues2017: CrossSubject ---
paradigm_rodrigues = RestingStateToP300Adapter(events=dict(closed=1, open=2))
evaluation_rodrigues = TLCrossSubjectEvaluation(
    paradigm=paradigm_rodrigues,
    datasets=[Rodrigues2017()],
    suffix="nch_study_rodrigues",
    overwrite=overwrite,
    n_splits=3,
    random_state=seed,
)
results_rodrigues = evaluation_rodrigues.process(pipelines)

# --- Hinss2021: CrossSubject ---
paradigm_hinss = RestingStateToP300Adapter(
    events=dict(easy=2, medium=3), tmin=0, tmax=0.5
)
evaluation_hinss_cs = TLCrossSubjectEvaluation(
    paradigm=paradigm_hinss,
    datasets=[Hinss2021()],
    suffix="nch_study_hinss_cs",
    overwrite=overwrite,
    n_splits=3,
    random_state=seed,
)
results_hinss_cs = evaluation_hinss_cs.process(pipelines)

##############################################################################
# Aggregate Results
# -----------------

results = pd.concat(
    [results_cattan, results_rodrigues, results_hinss_cs],
    ignore_index=True,
)

print(results)
print("Averaging the session performance:")
print(results.groupby("pipeline")[["score", "time"]].mean())

preferred_order = [
    "NCH+MIN_HULL_QAOACV(Ulvi)",
    "NCH+MIN_HULL_NAIVEQAOA",
    "MDWM(0.5)",
    "TS+LDA+TL",
    "MDM+TL",
    "TS+LDA",
    "MDM",
]
active_pipelines = results["pipeline"].unique()
print(active_pipelines)
order = [p for p in preferred_order if p in active_pipelines]

##############################################################################
# Plot Results: Raw Scores
# ------------------------
#
# Strip + point plot of per-subject scores across all datasets.

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

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
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##############################################################################
# Plot Results: Statistical Analysis
# ------------------------------------
#
# Summary plot showing pairwise statistical significance across datasets.
# Green = significantly higher, grey = no significant difference,
# red = significantly lower.

stats = compute_dataset_statistics(results)
P, T = find_significant_differences(stats)

fig_stat = summary_plot(P, T)
plt.tight_layout()
plt.show()

##############################################################################
# Plot Results: Meta-Analysis
# ---------------------------
#
# Standardized effect sizes with confidence intervals across datasets,
# one plot per pair of pipelines (alg1 is hypothesized to outperform alg2).

for alg1, alg2 in itertools.combinations(order, 2):
    fig_meta = meta_analysis_plot(stats, alg1, alg2)
    plt.tight_layout()
    plt.show()
