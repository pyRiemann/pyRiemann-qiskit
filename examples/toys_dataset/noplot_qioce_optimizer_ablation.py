"""
====================================================================
Optimizer ablation study for ContinuousQIOCEClassifier
====================================================================

Comparison of six optimizers for training ContinuousQIOCEClassifier
on a toy binary classification dataset:

- **L-BFGS-B**: quasi-Newton gradient method with bounds (default)
- **SLSQP**: sequential least-squares programming, gradient-based with bounds
- **COBYLA**: gradient-free, derivative-free linear approximation
- **SPSA**: stochastic perturbation gradient approximation
- **NFT**: Nakanishi-Fujii-Todo, quantum-native parameter-shift method
- **Anderson**: Anderson acceleration on the Riemannian manifold

Three metrics are reported across cross-validation folds:

- ROC AUC
- Training time
- Number of loss function evaluations (cost of optimization)

A convergence curve (loss vs. function evaluation) is also shown,
averaged across folds.

"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import time

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from qiskit_algorithms.optimizers import COBYLA, NFT, SLSQP, SPSA, L_BFGS_B
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pyriemann_qiskit.classification import ContinuousQIOCEClassifier
from pyriemann_qiskit.utils.anderson_optimizer import AndersonAccelerationOptimizer

print(__doc__)

###############################################################################
# Dataset
# -------
#
# Small binary classification problem with 4 features so that the QAOA
# circuit stays tractable (2^4 = 16 amplitudes).

seed = 42
n_features = 4
n_samples = 60
n_splits = 3

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=3,
    n_redundant=1,
    flip_y=0.05,
    random_state=seed,
)
X = StandardScaler().fit_transform(X)

###############################################################################
# Optimizers
# ----------
#
# All optimizers are given a comparable budget.
# Anderson, SPSA and NFT run for 25 iterations; L-BFGS-B, SLSQP and COBYLA
# are capped at 100 iterations / 200 function evaluations to match their
# typical usage in the main study.
# SLSQP and L-BFGS-B both handle bounds natively and compute numerical
# gradients when jac=None. NFT uses the parameter-shift rule, making it
# quantum-native without requiring an explicit gradient function.

optimizer_configs = [
    ("L-BFGS-B", L_BFGS_B(maxiter=100, maxfun=200)),
    ("SLSQP", SLSQP(maxiter=100)),
    ("COBYLA", COBYLA(maxiter=100)),
    ("SPSA", SPSA(maxiter=25)),
    ("NFT", NFT(maxiter=25)),
    ("Anderson", AndersonAccelerationOptimizer(maxiter=25)),
]

###############################################################################
# Cross-validated evaluation
# --------------------------

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

results = {
    name: {"acc": [], "time": [], "nfev": [], "loss_curves": []}
    for name, _ in optimizer_configs
}

for name, opt in optimizer_configs:
    print(f"\n=== {name} ===")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = ContinuousQIOCEClassifier(
            n_reps=2,
            max_features=n_features,
            optimizer=opt,
            random_state=seed + fold,
        )

        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        acc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        # training_loss_history_ has one entry per loss() call,
        # so its length equals the number of function evaluations.
        nfev = len(clf.training_loss_history_)
        loss_curve = clf.training_loss_history_

        results[name]["acc"].append(acc)
        results[name]["time"].append(elapsed)
        results[name]["nfev"].append(nfev)
        results[name]["loss_curves"].append(loss_curve)

        print(f"  Fold {fold + 1}: auc={acc:.3f}, nfev={nfev}, time={elapsed:.1f}s")

###############################################################################
# Plots
# -----

names = [name for name, _ in optimizer_configs]
colors = ["#4C72B0", "#9467BD", "#DD8452", "#55A868", "#8C564B", "#C44E52"]
x_pos = np.arange(len(names))
width = 0.5

fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="white")
fig.suptitle("QIOCE optimizer ablation — toy dataset", fontsize=13)

# --- Accuracy ---
ax = axes[0]
means = [np.mean(results[n]["acc"]) for n in names]
stds = [np.std(results[n]["acc"]) for n in names]
bars = ax.bar(x_pos, means, width, yerr=stds, capsize=5, color=colors, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("ROC AUC")
ax.set_title("ROC AUC")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.legend(fontsize=8)

# --- Training time ---
ax = axes[1]
means = [np.mean(results[n]["time"]) for n in names]
stds = [np.std(results[n]["time"]) for n in names]
ax.bar(x_pos, means, width, yerr=stds, capsize=5, color=colors, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Time (s)")
ax.set_title("Training time")

# --- Function evaluations ---
ax = axes[2]
means = [np.mean(results[n]["nfev"]) for n in names]
stds = [np.std(results[n]["nfev"]) for n in names]
ax.bar(x_pos, means, width, yerr=stds, capsize=5, color=colors, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("# loss evaluations")
ax.set_title("Optimization cost")

plt.tight_layout()
plt.show()

###############################################################################
# Convergence curves
# ------------------
#
# Average loss per function evaluation, truncated to the shortest curve
# across folds so that averaging is well-defined.

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")

for (name, _), color in zip(optimizer_configs, colors):
    curves = results[name]["loss_curves"]
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    x_eval = np.arange(1, min_len + 1)
    ax.plot(x_eval, mean_curve, label=name, color=color, linewidth=2)
    ax.fill_between(
        x_eval,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15,
        color=color,
    )

ax.set_xlabel("Function evaluation")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Convergence (mean ± std across folds)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
