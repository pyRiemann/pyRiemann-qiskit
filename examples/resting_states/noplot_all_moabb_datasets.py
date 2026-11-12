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
- QuantumStateDiscriminator operates directly on raw EEG epochs. Its TL
  variant receives the aligned covariance as its matrix square root S, so
  that S @ S.T rebuilds the covariance it would otherwise compute.
- TL variants align every subject on its own trials, without labels. At
  test time each session is scored as one batch, so a test subject with two
  sessions (Hinss2021) is aligned per session.

"""

# Author: Gregoire Cattan
# Modified from noplot_nch_study.py
# License: BSD (3-clause)

import argparse
import random

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
from pyriemann.transfer import MDWM
from pyriemann.utils.base import sqrtm
from qiskit_algorithms.optimizers import NFT
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import (
    ContinuousQIOCEClassifier,
    QuanticNCH,
    QuantumStateDiscriminator,
)
from pyriemann_qiskit.pipelines import QuantumClassifierWithDefaultRiemannianPipeline
from pyriemann_qiskit.utils.hyper_params_factory import (
    create_mixer_with_circular_entanglement,
)
from pyriemann_qiskit.utils.transfer import (
    Adapter,
    SelfAlignAdapter,
    TLCrossSubjectEvaluation,
)

print(__doc__)

set_log_level("info")

##############################################################################
# Set global seed for better reproducibility

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--only", default="", help="comma-separated pipeline names")
parser.add_argument(
    "--datasets", default="cattan,rodrigues,hinss", help="comma-separated datasets"
)
args, _ = parser.parse_known_args()
wanted = args.datasets.split(",")
seed = args.seed
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

pipelines["CSP+MDM"] = make_pipeline(
    sf,
    CSP(nfilter=8, log=False),
    MDM(),
)

pipelines["MDM"] = make_pipeline(
    sf,
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

# n_jobs=1: scoring is a trace per trial, and worker start-up would dominate
# the measured prediction time.
pipelines["QuantumStateDiscriminator"] = make_pipeline(
    QuantumStateDiscriminator(n_jobs=1),
)

# Quantum-kernel baseline: the library's default Riemannian pipeline
# (XdawnCovariances, tangent space, PCA, QuanticSVM with an X feature map),
# with 5 components (5 qubits) as in the library's examples. It takes raw
# epochs. Not run on Hinss2021: the kernel over 8,344 training trials takes
# hours per fold.
pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    dim_red=PCA(n_components=5),
    params={"seed": seed, "verbose": False},
)

pipelines["QIOCE"] = make_pipeline(
    sf,
    Whitening(dim_red={"n_components": 4}),
    TangentSpace(metric="riemann"),
    ContinuousQIOCEClassifier(
        n_reps=2, max_features=10, optimizer=NFT(maxiter=25), random_state=seed
    ),
)

pipelines["MDM+TL"] = SelfAlignAdapter(preprocessing=sf, estimator=MDM())

# Kept as in the original benchmark. MDWM needs labelled target data, so its
# target domain here is the first training subject (TLCrossSubjectEvaluation
# passes it), never the held-out subject.
pipelines["MDWM(0.5)"] = Adapter(
    preprocessing=sf,
    estimator=MDWM(domain_tradeoff=0.5, target_domain=None, metric="riemann"),
)


class CovSqrt(TransformerMixin, BaseEstimator):
    """Matrix square root S of each covariance, so that S @ S.T gives it back."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sqrtm(X)


pipelines["QuantumStateDiscriminator+TL"] = SelfAlignAdapter(
    preprocessing=sf,
    estimator=make_pipeline(CovSqrt(), QuantumStateDiscriminator(n_jobs=1)),
)

pipelines["TS+LR+TL"] = SelfAlignAdapter(
    preprocessing=sf,
    estimator=make_pipeline(TangentSpace(metric="riemann"), LR()),
)

pipelines["NCH+MIN_HULL_QAOACV(Ulvi)+TL"] = SelfAlignAdapter(
    preprocessing=sf,
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
)

pipelines["NCH+MIN_HULL_NAIVEQAOA+TL"] = SelfAlignAdapter(
    preprocessing=sf,
    estimator=QuanticNCH(
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        n_reps=2,
    ),
)

# Dimension reduction happens before alignment: once centered, the mean of the
# matrices is close to the identity and no longer ranks any direction.
pipelines["QIOCE+TL"] = SelfAlignAdapter(
    preprocessing=make_pipeline(
        Covariances(estimator="lwf"),
        Whitening(dim_red={"n_components": 4}),
    ),
    estimator=make_pipeline(
        TangentSpace(metric="riemann"),
        ContinuousQIOCEClassifier(
            n_reps=2, max_features=10, optimizer=NFT(maxiter=25), random_state=seed
        ),
    ),
)

if args.only:
    pipelines = {k: pipelines[k] for k in args.only.split(",")}

print("Total pipelines to evaluate: ", len(pipelines))

overwrite = False  # set to True if we want to overwrite cached results
common = dict(
    overwrite=overwrite,
    n_splits=None,
    random_state=seed,
    additional_columns=["score_time"],
)
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
    suffix=f"nch_study_cattan_seed{seed}",
    **common,
)
parts = []
if "cattan" in wanted:
    parts.append(evaluation_cattan.process(pipelines))

# --- Rodrigues2017: CrossSubject ---
# Blocks last 10 s and alternate, so the epoch must stay inside its own block.
# The adapter's default window (10-50 s) starts after the block has ended.
# 2-8 s is the window of the dataset authors' example, away from block edges.
paradigm_rodrigues = RestingStateToP300Adapter(
    events=dict(closed=1, open=2), tmin=2, tmax=8
)
evaluation_rodrigues = TLCrossSubjectEvaluation(
    paradigm=paradigm_rodrigues,
    datasets=[Rodrigues2017()],
    suffix=f"nch_study_rodrigues_w2-8_seed{seed}",
    **common,
)
if "rodrigues" in wanted:
    parts.append(evaluation_rodrigues.process(pipelines))

# --- Hinss2021: CrossSubject ---
paradigm_hinss = RestingStateToP300Adapter(
    events=dict(easy=2, medium=3), tmin=0, tmax=0.5
)
evaluation_hinss_cs = TLCrossSubjectEvaluation(
    paradigm=paradigm_hinss,
    datasets=[Hinss2021()],
    suffix=f"nch_study_hinss_cs_seed{seed}",
    **common,
)
if "hinss" in wanted:
    parts.append(evaluation_hinss_cs.process(pipelines))

##############################################################################
# Aggregate Results
# -----------------

results = pd.concat(parts, ignore_index=True)

print(results)
print("Averaging the session performance:")
print(results.groupby("pipeline")[["score", "time", "score_time"]].mean())
print(results.groupby(["dataset", "pipeline"])[["score", "time", "score_time"]].mean())
results.to_csv(f"results_seed{seed}.csv", index=False)

preferred_order = [
    "NCH+MIN_HULL_QAOACV(Ulvi)+TL",
    "NCH+MIN_HULL_QAOACV(Ulvi)",
    "NCH+MIN_HULL_NAIVEQAOA",
    "NCH+MIN_HULL_NAIVEQAOA+TL",
    "QIOCE",
    "QIOCE+TL",
    "RG+QuantumSVM",
    "QuantumStateDiscriminator",
    "QuantumStateDiscriminator+TL",
    "MDWM(0.5)",
    "MDM+TL",
    "MDM",
    "CSP+MDM",
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

top3 = ["CSP+MDM", "QuantumStateDiscriminator", "QIOCE"]
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
