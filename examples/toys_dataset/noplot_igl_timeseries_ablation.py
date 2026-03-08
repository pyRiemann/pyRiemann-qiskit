"""
====================================================================
IGL Time-Series ablation study on toy EEG data
====================================================================

Grid-search over kernel operators, training epochs, and Tikhonov
regularization (``source_l2``) for ``IGLTimeSeriesSklearnClassifier``
on synthetically generated EEG epochs.

The ablation factors are:

- **operator**: gaussian, cauchy, helmholtz, gabor, laplacian, mexican_hat
- **epochs**: 1000, 1500
- **source_l2** (Tikhonov lambda for the lstsq W_out solve): 0.01, 0.1

Data generation mirrors the toy EEG setup used in other ablation
scripts: each class has a distinct channel activation pattern,
simulating an ERP-like paradigm (2 classes, single subject).
"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
n_channels = 8
n_times = 100
n_classes = 2
n_splits = 5

# Ablation grid
operators = ["gaussian", "cauchy", "helmholtz", "gabor", "laplacian", "mexican_hat"]
epochs_list = [1000, 1500]
lambdas = [0.01, 0.1]

# Fixed IGL hyper-params (not ablated)
n_components = 16
n_anchors = 64
n_scales = 3
warmup_fraction = 0.2  # warmup_epochs = epochs * warmup_fraction

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
# Ablation loop
# -------------
#
# For each combination of (operator, epochs, source_l2) we run
# StratifiedKFold cross-validation and collect per-fold ROC AUC.

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

results = []
grid = list(itertools.product(operators, epochs_list, lambdas))
print(f"\nTotal configurations: {len(grid)}  |  CV folds: {n_splits}")
print(f"Total fits: {len(grid) * n_splits}\n")

for op, epochs, lam in grid:
    warmup_epochs = max(1, int(epochs * warmup_fraction))
    config = VPConfig(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        source_l2=lam,
        log_every=epochs,   # only log at the end to keep output clean
        verbose=False,
    )
    clf = IGLTimeSeriesSklearnClassifier(
        n_components=n_components,
        n_anchors=n_anchors,
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
        proba = clf_.predict_proba(X[test_idx])
        auc = roc_auc_score(y[test_idx], proba[:, 1])
        fold_scores.append(auc)
        results.append(
            {
                "operator": op,
                "epochs": epochs,
                "source_l2": lam,
                "fold": fold,
                "auc": auc,
            }
        )

    mean_auc = np.mean(fold_scores)
    print(
        f"op={op:12s}  epochs={epochs:4d}  lambda={lam:.2f}  "
        f"mean_auc={mean_auc:.3f}  folds={[f'{s:.3f}' for s in fold_scores]}"
    )

results = pd.DataFrame(results)

##############################################################################
# Summary table
# -------------

summary = (
    results.groupby(["operator", "epochs", "source_l2"])["auc"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
print("\n=== Ablation Summary (sorted by mean AUC) ===")
print(summary.to_string(index=False))

##############################################################################
# Plots
# -----
#
# 1. Operator comparison (averaged over epochs and lambda).
# 2. Epochs × lambda heatmap (averaged over operators).

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

# --- Plot 1: operator comparison ---
ax = axes[0]
op_summary = results.groupby("operator")["auc"].mean().reset_index().sort_values(
    "auc", ascending=False
)
sns.barplot(data=results, x="operator", y="auc", order=op_summary["operator"], ax=ax,
            capsize=0.1, palette="Set2", errorbar="sd")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.set_title("Operator comparison\n(averaged over epochs & λ)")
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Kernel operator")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=8)

# --- Plot 2: epochs × lambda heatmap ---
ax = axes[1]
pivot = (
    results.groupby(["epochs", "source_l2"])["auc"]
    .mean()
    .unstack("source_l2")
)
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    vmin=0.5,
    vmax=1.0,
    ax=ax,
    cbar_kws={"label": "mean ROC AUC"},
)
ax.set_title("Epochs × λ heatmap\n(averaged over operators)")
ax.set_xlabel("source_l2 (λ)")
ax.set_ylabel("epochs")

plt.suptitle("IGLTimeSeriesSklearnClassifier — Ablation on toy EEG", fontsize=13)
plt.tight_layout()
plt.show()

##############################################################################
# Per-operator detail: strip + point plot faceted by (epochs, lambda)

g = sns.FacetGrid(
    results,
    col="epochs",
    row="source_l2",
    height=3.5,
    aspect=1.6,
    sharey=True,
)
g.map_dataframe(
    sns.stripplot,
    x="operator",
    y="auc",
    jitter=True,
    alpha=0.5,
    palette="Set1",
    order=op_summary["operator"].tolist(),
)
g.map_dataframe(
    sns.pointplot,
    x="operator",
    y="auc",
    palette="Set1",
    order=op_summary["operator"].tolist(),
    linestyle="none",
)
for ax in g.axes.flat:
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.tick_params(axis="x", rotation=30)
g.set_axis_labels("Kernel operator", "ROC AUC")
g.set_titles(col_template="epochs={col_name}", row_template="λ={row_name}")
g.figure.suptitle(
    "IGL ablation — per (epochs, λ) breakdown", fontsize=12, y=1.02
)
plt.tight_layout()
plt.show()
