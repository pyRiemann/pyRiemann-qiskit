"""
====================================================================
QAOA circuit depth ablation study — n_reps vs AUC and training time
====================================================================

This study answers: **how deep should the QAOA circuit be?**
Does increasing ``n_reps`` (number of QAOA layers) improve ROC AUC,
and at what computational cost?

Three quantum pipelines are benchmarked across ``n_reps ∈ {2, 3, 4, 5}``:

- **NCH (Naive QAOA)**: QuanticNCH with default mixer
- **NCH (Ulvi)**: QuanticNCH with circular-entanglement mixer + angle encoding
- **QIOCE**: ContinuousQIOCEClassifier (ULVI-style angle optimizer)

All use ``quantum=True`` (Aer simulator) to meaningfully vary circuit depth.
Metrics: ROC AUC + training time per fold — so the accuracy/cost trade-off
is visible.

"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import ContinuousQIOCEClassifier, QuanticNCH
from pyriemann_qiskit.utils.hyper_params_factory import (
    create_mixer_with_circular_entanglement,
)

print(__doc__)

##############################################################################
# Parameters

seed = 42
n_channels = 4
n_times = 50
n_classes = 2
n_trials_per_class = 30

##############################################################################
# Data Generation
# ---------------
#
# Single-subject synthetic EEG-like data with class-specific channel
# activations. The Cholesky mixing matrix ensures positive-definite
# covariance matrices and linear separability between classes.
#
# Classes differ in VARIANCE (channel scale), not mean. Covariance estimators
# like "lwf" center the data (subtract the mean), so a mean shift produces
# identical covariance matrices across classes — AUC = 0.5 regardless of
# signal amplitude. Scaling the noise instead preserves the class signal
# through the centering step:
#   class k → channel k has 5× higher std → covariance differs from other class.


def make_subject_data(n_trials_per_class, n_channels, n_times, n_classes, subj_seed):
    rng = np.random.RandomState(subj_seed)
    M = rng.randn(n_channels, n_channels)
    A = np.linalg.cholesky(M @ M.T + n_channels * np.eye(n_channels))
    X_list, y_list = [], []
    for cls in range(n_classes):
        scale = np.ones(n_channels)
        scale[cls] = 5.0  # class k: channel k has 5× higher variance
        noise = rng.randn(n_trials_per_class, n_channels, n_times)
        noise *= scale[:, None]  # scale per channel — survives covariance centering
        X_cls = np.einsum("ij,tjk->tik", A, noise)
        X_list.append(X_cls)
        y_list.append(np.full(n_trials_per_class, cls))
    return np.concatenate(X_list), np.concatenate(y_list)


X, y = make_subject_data(n_trials_per_class, n_channels, n_times, n_classes,
                         subj_seed=seed)
print(f"Dataset: X={X.shape}, y={y.shape}")

##############################################################################
# Pipelines
# ---------
#
# Pipelines are rebuilt for each n_reps value so the circuit depth parameter
# is applied correctly. max_features=10 for QIOCE matches the tangent-space
# output dimension for 4 whitened components: 4×5/2 = 10.

n_reps_range = [2, 3, 4, 5]


def make_pipelines(n_reps, seed):
    return {
        "NCH (Naive QAOA)": make_pipeline(
            Covariances(estimator="lwf"),
            QuanticNCH(
                quantum=True,
                n_reps=n_reps,
                create_mixer=None,
                n_samples_per_hull=2,
                subsampling="min",
                shots=100,
                seed=seed,
            ),
        ),
        "NCH (Ulvi)": make_pipeline(
            Covariances(estimator="lwf"),
            QuanticNCH(
                quantum=True,
                n_reps=n_reps,
                create_mixer=create_mixer_with_circular_entanglement(0),
                qaoacv_implementation="ulvi",
                shots=100,
                n_samples_per_hull=2,
                subsampling="min",
                seed=seed,
            ),
        ),
        "QIOCE": make_pipeline(
            Covariances(estimator="lwf"),
            Whitening(dim_red={"n_components": 4}),
            TangentSpace(metric="riemann"),
            ContinuousQIOCEClassifier(n_reps=n_reps, max_features=10),
        ),
    }


##############################################################################
# Evaluation
# ----------
#
# StratifiedKFold with 3 splits. For each (n_reps, pipeline, fold) triple,
# we record ROC AUC and training time.

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

results = []
for n_reps in n_reps_range:
    print(f"\n=== n_reps={n_reps} ===")
    pipelines = make_pipelines(n_reps, seed)
    for name, clf in pipelines.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            clf_ = deepcopy(clf)
            t0 = time.time()
            clf_.fit(X[train_idx], y[train_idx])
            elapsed = time.time() - t0
            auc = roc_auc_score(y[test_idx], clf_.predict_proba(X[test_idx])[:, 1])
            results.append({
                "pipeline": name, "n_reps": n_reps,
                "fold": fold, "auc": auc, "time": elapsed,
            })
            print(f"  {name} fold={fold}: auc={auc:.3f}, time={elapsed:.1f}s")

results = pd.DataFrame(results)
print(f"\nTotal rows: {len(results)}")
print("\nMean AUC per pipeline × n_reps:")
print(results.groupby(["pipeline", "n_reps"])[["auc"]].mean().round(3))

##############################################################################
# Plot Results
# ------------
#
# Two subplots side by side: classification performance (AUC) and
# computational cost (training time). One line per pipeline, shaded
# band = ±1 std across folds.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
fig.suptitle("QAOA circuit depth ablation (n_reps) — quantum simulation", fontsize=13)

pipeline_names = ["NCH (Naive QAOA)", "NCH (Ulvi)", "QIOCE"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

for ax_idx, (metric, ylabel, title) in enumerate([
    ("auc",  "ROC AUC",           "Classification performance"),
    ("time", "Training time (s)", "Computational cost"),
]):
    ax = axes[ax_idx]
    for name, color in zip(pipeline_names, colors):
        sub = results[results["pipeline"] == name]
        means = sub.groupby("n_reps")[metric].mean()
        stds = sub.groupby("n_reps")[metric].std()
        ax.plot(means.index, means.values, marker="o", label=name, color=color)
        ax.fill_between(
            means.index,
            means.values - stds.values,
            means.values + stds.values,
            alpha=0.15,
            color=color,
        )
    if metric == "auc":
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("n_reps")
    ax.set_xticks(n_reps_range)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
