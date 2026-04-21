"""
====================================================================
Classification of resting-state datasets from MOABB using NCH, QIOCE,
and QuantumStateDiscriminator
====================================================================

Comparison of classical pipelines (MDM, TS+LR) and quantum pipelines
(NCH, QIOCE, QuantumStateDiscriminator) across three resting-state
datasets: Hinss2021 (CrossSubject), Rodrigues2017 (CrossSubject), and
Cattan2019_PHMD (CrossSubject). Both plain and transfer-learning (TL)
variants are benchmarked where applicable. Results are aggregated and
analyzed with MOABB statistical tools.

Notes:
- QuantumStateDiscriminator operates directly on raw EEG epochs (no
  covariance estimation) and is evaluated without a TL variant.

"""
# Author: Gregoire Cattan
# Modified from noplot_nch_study.py
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
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import MDWM, TLCenter, TLClassifier, TLRotate, TLScale
from qiskit_algorithms.optimizers import NFT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import (
    ContinuousQIOCEClassifier,
    QuanticNCH,
    QuantumStateDiscriminator,
)
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
)

n_samples_per_hull = 6

pipelines = {}

pipelines["MDM"] = make_pipeline(
    sf,
    CSP(nfilter=8, log=False),
    MDM(),
)

pipelines["TS+LR"] = make_pipeline(
    sf,
    TangentSpace(metric="riemann"),
    LR(),
)

pipelines["NCH+MIN_HULL_QAOACV(Ulvi)"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        create_mixer=create_mixer_with_circular_entanglement(0),
        shots=100,
        qaoa_optimizer=NFT(maxiter=25),
        n_reps=2,
        qaoacv_implementation="ulvi",
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
        n_reps=2,
    ),
)

pipelines["QuantumStateDiscriminator"] = make_pipeline(
    QuantumStateDiscriminator(n_jobs=12),
)

pipelines["QIOCE"] = make_pipeline(
    sf,
    Whitening(dim_red={"n_components": 4}),
    TangentSpace(metric="riemann"),
    ContinuousQIOCEClassifier(n_reps=2, max_features=10, optimizer=NFT(maxiter=25)),
)

pipelines["MDM+TL"] = Adapter(
    preprocessing=sf,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(target_domain=None, estimator=MDM(), domain_weight=None),
    ),
)

pipelines["TS+LR+TL"] = Adapter(
    preprocessing=sf,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(
            target_domain=None,
            estimator=make_pipeline(TangentSpace(metric="riemann"), LR()),
            domain_weight=None,
        ),
    ),
)

pipelines["MDWM(0.5)"] = Adapter(
    preprocessing=sf,
    estimator=MDWM(domain_tradeoff=0.5, target_domain=None, metric="riemann"),
)

pipelines["NCH+MIN_HULL_QAOACV(Ulvi)+TL"] = Adapter(
    preprocessing=sf,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(
            target_domain=None,
            estimator=QuanticNCH(
                seed=seed,
                n_samples_per_hull=n_samples_per_hull,
                n_jobs=12,
                subsampling="min",
                quantum=True,
                create_mixer=create_mixer_with_circular_entanglement(0),
                shots=100,
                qaoa_optimizer=NFT(maxiter=25),
                n_reps=2,
                qaoacv_implementation="ulvi",
            ),
            domain_weight=None,
        ),
    ),
)

pipelines["NCH+MIN_HULL_NAIVEQAOA+TL"] = Adapter(
    preprocessing=sf,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(
            target_domain=None,
            estimator=QuanticNCH(
                seed=seed,
                n_samples_per_hull=n_samples_per_hull,
                n_jobs=12,
                subsampling="min",
                quantum=True,
                n_reps=2,
            ),
            domain_weight=None,
        ),
    ),
)

pipelines["QIOCE+TL"] = Adapter(
    preprocessing=sf,
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(
            target_domain=None,
            estimator=make_pipeline(
                Whitening(dim_red={"n_components": 4}),
                TangentSpace(metric="riemann"),
                ContinuousQIOCEClassifier(
                    n_reps=2, max_features=10, optimizer=NFT(maxiter=25)
                ),
            ),
            domain_weight=None,
        ),
    ),
)

print("Total pipelines to evaluate: ", len(pipelines))

overwrite = False  # set to True if we want to overwrite cached results
N_SPLITS = None
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
    n_splits=N_SPLITS,
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
    n_splits=N_SPLITS,
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
    n_splits=N_SPLITS,
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
    "NCH+MIN_HULL_QAOACV(Ulvi)+TL",
    "NCH+MIN_HULL_QAOACV(Ulvi)",
    "NCH+MIN_HULL_NAIVEQAOA",
    "NCH+MIN_HULL_NAIVEQAOA+TL",
    "QIOCE",
    "QIOCE+TL",
    "QuantumStateDiscriminator",
    "MDWM(0.5)",
    "MDM+TL",
    "MDM",
    "TS+LR",
    "TS+LR+TL",
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
# Plot Results: Per-Dataset Comparison (Top 3 Pipelines)
# -------------------------------------------------------
#
# Strip + point plot of per-subject scores per dataset for the three
# main performers: MDM, QuantumStateDiscriminator, and QIOCE.

top3 = ["MDM", "QuantumStateDiscriminator", "QIOCE"]
results_top3 = results[results["pipeline"].isin(top3)]

fig2, ax2 = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results_top3,
    y="score",
    x="dataset",
    hue="pipeline",
    ax=ax2,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
    hue_order=top3,
    dodge=True,
)
sns.pointplot(
    data=results_top3,
    y="score",
    x="dataset",
    hue="pipeline",
    ax=ax2,
    palette="Set1",
    hue_order=top3,
    dodge=True,
)

ax2.set_ylabel("ROC AUC")
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[: len(top3)], labels[: len(top3)], title="Pipeline")
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
