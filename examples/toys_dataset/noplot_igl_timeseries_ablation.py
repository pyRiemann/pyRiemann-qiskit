"""
====================================================================
IGL Time-Series ablation study on toy EEG data
====================================================================

Grid-search over kernel operators, Tikhonov regularization
(``source_l2``), spatial filter rank (``n_components``), and temporal
basis size (``n_anchors``) for ``IGLTimeSeriesSklearnClassifier`` on
synthetically generated EEG epochs.

The ablation factors are:

- **operator**: gaussian, cauchy, helmholtz, gabor, laplacian, mexican_hat
- **source_l2**: 0.01, 0.1  (Tikhonov lambda for the exact W_out lstsq solve)
- **n_components**: 2, 4, 16   (spatial filter bank width, like CSP rank)
- **n_anchors**: 32, 64     (temporal basis expressivity)

A **LinearHead** baseline is also evaluated: mean-pool over time then
logistic regression — no kernel, no VP, pure Wx+b.  This lets us
measure how much the Green kernel temporal integration actually helps.

EEG is modelled as a mixture of sinusoids (delta, theta, alpha, beta,
gamma) with a spatial mixing matrix.  Class discrimination is encoded
in band amplitude: class 0 has boosted alpha power, class 1 has
boosted beta power.
"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyriemann_qiskit.classification.igl_reference import (
    IGLTimeSeriesSklearnClassifier,
    VPConfig,
)

print(__doc__)

##############################################################################
# Parameters

seed = 42
rng = np.random.RandomState(seed)

n_trials_per_class = 30
n_channels = 16
n_times = 512
n_classes = 2
n_splits = 3

# Ablation grid
operators = ["gaussian", "cauchy", "helmholtz", "gabor", "laplacian", "mexican_hat"]
lambdas = [0.01, 0.1]
n_components_list = [2, 4, 16]   # spatial filter rank (sparsity in channel space)
n_anchors_list = [32, 64]     # temporal basis size (expressivity)

# Fixed IGL hyper-params (not ablated)
n_scales = 3
warmup_fraction = 0.2  # warmup_epochs = epochs * warmup_fraction
epochs = 1500

##############################################################################
# Data Generation
# ---------------
#
# EEG is modelled as a mixture of sinusoids across canonical frequency bands,
# with additive Gaussian noise.  The class label is encoded in the *amplitude*
# of one frequency band per class:
#
#   Class 0: elevated alpha power  (8–13 Hz)
#   Class 1: elevated beta power   (13–30 Hz)
#
# All channels share the same mixture; a random spatial mixing matrix
# simulates volume conduction so that no single channel is trivially
# discriminative.
#
# Band centre frequencies (Hz) and their default amplitudes:
#   delta  ~2 Hz  → 0.5   (background slow drift)
#   theta  ~6 Hz  → 0.5
#   alpha ~10 Hz  → 1.0   (boosted for class 0)
#   beta  ~20 Hz  → 1.0   (boosted for class 1)
#   gamma ~40 Hz  → 0.3   (high-freq noise-like component)

sfreq = 256  # sampling frequency (Hz) — sets the physical time axis


def make_eeg_data(n_trials_per_class, n_channels, n_times, n_classes, rng, sfreq=256):
    t = np.linspace(0, n_times / sfreq, n_times, endpoint=False)  # time axis in seconds

    # Canonical EEG bands: (centre_freq_Hz, base_amplitude)
    bands = [
        (2.0,  0.5),   # delta
        (6.0,  0.5),   # theta
        (10.0, 1.0),   # alpha
        (20.0, 1.0),   # beta
        (40.0, 0.3),   # gamma
    ]
    alpha_idx = 2  # index of the alpha band in `bands`
    beta_idx  = 3  # index of the beta  band in `bands`

    # Class 0 boosts alpha; class 1 boosts beta
    class_boost = {0: (alpha_idx, 2.5), 1: (beta_idx, 2.5)}

    # Random spatial mixing matrix shared across all trials (volume conduction)
    M = rng.randn(n_channels, n_channels)
    A = np.linalg.cholesky(M @ M.T + n_channels * np.eye(n_channels))
    A /= np.linalg.norm(A, axis=0, keepdims=True)  # normalise columns

    X_list, y_list = [], []
    for cls in range(n_classes):
        amplitudes = [amp for _, amp in bands]
        boost_idx, boost_val = class_boost[cls]
        amplitudes[boost_idx] = boost_val

        trials = []
        for _ in range(n_trials_per_class):
            # Build a single-channel template as a sum of sinusoids
            # with random per-trial phase offsets (trial-to-trial variability)
            template = np.zeros(n_times)
            for (freq, _), amp in zip(bands, amplitudes):
                phase = rng.uniform(0, 2 * np.pi)
                template += amp * np.cos(2 * np.pi * freq * t + phase)

            # Replicate to n_channels and apply spatial mixing + noise
            X_trial = np.tile(template, (n_channels, 1))          # [C, T]
            X_trial = A @ X_trial                                   # spatial mix
            X_trial += 0.3 * rng.randn(n_channels, n_times)        # sensor noise
            trials.append(X_trial)

        X_list.append(np.stack(trials))                             # [N, C, T]
        y_list.append(np.full(n_trials_per_class, cls))
    return np.concatenate(X_list), np.concatenate(y_list)


X, y = make_eeg_data(n_trials_per_class, n_channels, n_times, n_classes, rng, sfreq=sfreq)
print(f"Dataset: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

##############################################################################
# Linear-head baseline
# --------------------
#
# Mean-pool raw EEG over time [N, C, T] → [N, C], then fit a logistic
# regression (Wx+b).  No kernel, no Variable Projection.
# This baseline isolates how much the Green kernel temporal integration
# contributes over a simple linear readout of mean channel activity.


class MeanPoolLinear(BaseEstimator, ClassifierMixin):
    """Mean-pool over time + logistic regression. No temporal kernel."""

    def fit(self, X, y):
        self.pipe_ = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        )
        self.pipe_.fit(X.mean(axis=-1), y)   # X: [N, C, T] → [N, C]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X.mean(axis=-1))


##############################################################################
# Ablation loop
# -------------
#
# Two loops:
#   1. LinearHead baseline  — no hyper-params, just CV.
#   2. IGL grid             — all combinations of (operator, lambda,
#                             n_components, n_anchors).

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
results = []

# --- Baseline ---
print("=== LinearHead baseline ===")
baseline = MeanPoolLinear()
fold_scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    b = deepcopy(baseline)
    b.fit(X[train_idx], y[train_idx])
    auc = roc_auc_score(y[test_idx], b.predict_proba(X[test_idx])[:, 1])
    fold_scores.append(auc)
    results.append({"model": "LinearHead", "operator": "—", "source_l2": None,
                    "n_components": None, "n_anchors": None, "fold": fold,
                    "auc": auc, "eff_dim": None})
print(f"  mean_auc={np.mean(fold_scores):.3f}  folds={[f'{s:.3f}' for s in fold_scores]}")

# --- IGL grid ---
igl_grid = list(itertools.product(operators, lambdas, n_components_list, n_anchors_list))
print(f"\n=== IGL grid: {len(igl_grid)} configs × {n_splits} folds = "
      f"{len(igl_grid) * n_splits} fits ===\n")

for op, lam, n_comp, n_anch in igl_grid:
    warmup_epochs = max(1, int(epochs * warmup_fraction))
    config = VPConfig(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        source_l2=lam,
        log_every=epochs,   # only log at the end to keep output clean
        verbose=False,
    )
    clf = IGLTimeSeriesSklearnClassifier(
        n_components=n_comp,
        n_anchors=n_anch,
        n_scales=n_scales,
        operator=op,
        training="vp",
        vp_config=config,
        random_state=seed,
    )

    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf_ = deepcopy(clf)
        clf_.fit(X[train_idx], y[train_idx])
        auc = roc_auc_score(y[test_idx], clf_.predict_proba(X[test_idx])[:, 1])
        ed = clf_.effective_dimension()
        fold_scores.append(auc)
        results.append({"model": "IGL", "operator": op, "source_l2": lam,
                        "n_components": n_comp, "n_anchors": n_anch,
                        "fold": fold, "auc": auc, "eff_dim": ed})

    print(f"op={op:12s}  λ={lam:.2f}  K={n_comp:2d}  R={n_anch:2d}  "
          f"mean={np.mean(fold_scores):.3f}  "
          f"folds={[f'{s:.3f}' for s in fold_scores]}")

results = pd.DataFrame(results)

##############################################################################
# Summary tables
# --------------

igl_results = results[results["model"] == "IGL"]
baseline_auc = results[results["model"] == "LinearHead"]["auc"].mean()

print(f"\nLinearHead baseline mean AUC: {baseline_auc:.3f}")

summary = (
    igl_results.groupby(["operator", "source_l2", "n_components", "n_anchors"])
    .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
         eff_dim_mean=("eff_dim", "mean"), eff_dim_std=("eff_dim", "std"))
    .reset_index()
    .sort_values("auc_mean", ascending=False)
)
print("\n=== IGL Ablation Summary (top 10) ===")
print(summary.head(10).to_string(index=False))

##############################################################################
# Plot 1 — IGL vs LinearHead
# --------------------------
#
# Best IGL config (per fold) vs LinearHead.

best_igl = (
    igl_results.groupby(["operator", "source_l2", "n_components", "n_anchors"])["auc"]
    .mean()
    .idxmax()
)
best_igl_label = (
    f"IGL best\n(op={best_igl[0]}, λ={best_igl[1]}, K={best_igl[2]}, R={best_igl[3]})"
)
best_igl_rows = igl_results[
    (igl_results["operator"] == best_igl[0])
    & (igl_results["source_l2"] == best_igl[1])
    & (igl_results["n_components"] == best_igl[2])
    & (igl_results["n_anchors"] == best_igl[3])
].copy()
best_igl_rows["label"] = best_igl_label

baseline_rows = results[results["model"] == "LinearHead"].copy()
baseline_rows["label"] = "LinearHead\n(mean-pool + LR)"

compare_df = pd.concat([best_igl_rows, baseline_rows], ignore_index=True)

fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
sns.stripplot(data=compare_df, x="label", y="auc", jitter=True, alpha=0.6,
              palette="Set1", ax=ax)
sns.pointplot(data=compare_df, x="label", y="auc", palette="Set1",
              linestyle="none", ax=ax)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.axhline(baseline_auc, color="steelblue", linestyle=":", linewidth=1.2,
           label=f"LinearHead mean ({baseline_auc:.3f})")
ax.set_title("Best IGL vs LinearHead baseline")
ax.set_ylabel("ROC AUC")
ax.set_xlabel("")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

##############################################################################
# Plot 2 — Operator comparison
# ----------------------------
#
# IGL operator effect, averaged over (lambda, n_components, n_anchors).
# Horizontal dashed line = LinearHead baseline.

op_order = (
    igl_results.groupby("operator")["auc"].mean()
    .sort_values(ascending=False)
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
sns.barplot(data=igl_results, x="operator", y="auc", order=op_order,
            capsize=0.1, palette="Set2", errorbar="sd", ax=ax)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.axhline(baseline_auc, color="steelblue", linestyle=":", linewidth=1.5,
           label=f"LinearHead ({baseline_auc:.3f})")
ax.set_title("Operator comparison (averaged over λ, n_components, n_anchors)")
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Kernel operator")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

##############################################################################
# Plot 3 — Hyperparameter heatmaps
# ---------------------------------
#
# Left:  operator × source_l2  (averaged over n_components, n_anchors)
# Right: n_components × n_anchors  (averaged over operator, source_l2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

# operator × lambda
pivot_op_lam = (
    igl_results.groupby(["operator", "source_l2"])["auc"]
    .mean()
    .unstack("source_l2")
    .reindex(op_order)
)
sns.heatmap(pivot_op_lam, annot=True, fmt=".3f", cmap="YlGnBu",
            vmin=0.5, vmax=1.0, ax=axes[0],
            cbar_kws={"label": "mean ROC AUC"})
axes[0].set_title("operator × λ\n(averaged over K, R)")
axes[0].set_xlabel("source_l2 (λ)")
axes[0].set_ylabel("operator")

# n_components × n_anchors
pivot_kR = (
    igl_results.groupby(["n_components", "n_anchors"])["auc"]
    .mean()
    .unstack("n_anchors")
)
sns.heatmap(pivot_kR, annot=True, fmt=".3f", cmap="YlGnBu",
            vmin=0.5, vmax=1.0, ax=axes[1],
            cbar_kws={"label": "mean ROC AUC"})
axes[1].set_title("n_components (K) × n_anchors (R)\n(averaged over operator, λ)")
axes[1].set_xlabel("n_anchors (R)")
axes[1].set_ylabel("n_components (K)")

plt.suptitle("IGLTimeSeriesSklearnClassifier — Hyperparameter ablation", fontsize=13)
plt.tight_layout()
plt.show()

##############################################################################
# Plot 4 — Per-operator FacetGrid (lambda × n_components)
# --------------------------------------------------------

facet_df = igl_results.copy()
facet_df["config"] = (
    "K=" + facet_df["n_components"].astype(str)
    + " R=" + facet_df["n_anchors"].astype(str)
)

g = sns.FacetGrid(facet_df, col="source_l2", row="n_components",
                  height=3.5, aspect=1.6, sharey=True)
g.map_dataframe(sns.stripplot, x="operator", y="auc", jitter=True,
                alpha=0.5, palette="Set1", order=op_order)
g.map_dataframe(sns.pointplot, x="operator", y="auc", palette="Set1",
                order=op_order, linestyle="none")
for ax in g.axes.flat:
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(baseline_auc, color="steelblue", linestyle=":", linewidth=1.2)
    ax.tick_params(axis="x", rotation=30)
g.set_axis_labels("Kernel operator", "ROC AUC")
g.set_titles(col_template="λ={col_name}", row_template="K={row_name}")
g.figure.suptitle(
    "IGL — operator per (λ, K)  [blue dotted = LinearHead baseline]",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.show()

##############################################################################
# Plot 5 — Effective dimension
# ----------------------------
#
# Left:  mean effective K per operator (averaged over λ, n_components, n_anchors).
# Right: mean effective K vs mean AUC scatter — does using more active filters
#        actually help?  Each point is one (operator, λ, K, R) configuration.

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

# --- Left: effective dim per operator ---
ax = axes[0]
ed_op = (
    igl_results.groupby("operator")["eff_dim"]
    .mean()
    .reset_index()
    .sort_values("eff_dim", ascending=False)
)
sns.barplot(data=igl_results, x="operator", y="eff_dim",
            order=ed_op["operator"], palette="Set2", errorbar="sd",
            capsize=0.1, ax=ax)
ax.set_title("Mean effective K per operator\n(averaged over λ, K, R)")
ax.set_ylabel("Effective spatial filters")
ax.set_xlabel("Kernel operator")
ax.tick_params(axis="x", rotation=30)

# --- Right: eff_dim vs AUC scatter ---
ax = axes[1]
config_summary = (
    igl_results.groupby(["operator", "source_l2", "n_components", "n_anchors"])
    .agg(auc_mean=("auc", "mean"), eff_dim_mean=("eff_dim", "mean"))
    .reset_index()
)
scatter = ax.scatter(
    config_summary["eff_dim_mean"],
    config_summary["auc_mean"],
    c=pd.Categorical(config_summary["operator"]).codes,
    cmap="tab10",
    alpha=0.7,
    s=60,
)
ax.axhline(baseline_auc, color="steelblue", linestyle=":", linewidth=1.2,
           label=f"LinearHead ({baseline_auc:.3f})")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.set_xlabel("Mean effective spatial filters")
ax.set_ylabel("Mean ROC AUC")
ax.set_title("Effective K vs AUC\n(each point = one config)")
# legend for operators
for code, op in enumerate(config_summary["operator"].unique()):
    ax.scatter([], [], c=[plt.cm.tab10(code / 10)], label=op, s=40)
ax.legend(fontsize=7, ncol=2)

plt.suptitle("IGLTimeSeriesSklearnClassifier — Effective spatial dimension",
             fontsize=13)
plt.tight_layout()
plt.show()

# Print per-config effective dimension alongside AUC
print("\n=== Effective dimension summary (top 10 by AUC) ===")
print(
    summary[["operator", "source_l2", "n_components", "n_anchors",
             "auc_mean", "auc_std", "eff_dim_mean", "eff_dim_std"]]
    .head(10)
    .to_string(index=False)
)
