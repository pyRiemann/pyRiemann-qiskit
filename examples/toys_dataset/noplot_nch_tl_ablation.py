"""
====================================================================
Toy dataset ablation study — NCH vs MDM with/without Transfer Learning
====================================================================

Fast, self-contained illustration comparing:

- Classical baseline (MDM) vs quantum-classical (NCH, ``quantum=False``)
- Each with and without the full Riemannian TL alignment stack
  (TLCenter → TLScale → TLRotate → TLClassifier)

Ten synthetic subjects are generated with subject-specific channel mixing
(domain shift). A manual leave-one-subject-out loop mimics
``TLCrossSubjectEvaluation`` without any MOABB dependency, making this
script runnable in seconds for documentation, CI, and quick sanity checks.

"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.preprocessing import Whitening
from pyriemann.transfer import TLCenter, TLClassifier, TLRotate, TLScale, encode_domains
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.utils.math import to_xyz
from pyriemann_qiskit.visualization.manifold import plot_manifold

from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils.transfer import Adapter

print(__doc__)

##############################################################################
# Parameters

seed = 42
n_subjects = 10
n_trials_per_class = 20
n_channels = 4
n_times = 50
n_classes = 2
n_samples_per_hull = 2

##############################################################################
# Data Generation
# ---------------
#
# Each subject has a random channel mixing matrix that simulates domain shift.
# Class-specific channel activations make the problem linearly separable
# within subjects. The Cholesky mixing matrix ensures cross-subject variability
# while keeping covariance matrices positive definite.


def make_subject_data(n_trials_per_class, n_channels, n_times, n_classes, subj_seed):
    rng = np.random.RandomState(subj_seed)
    # Per-subject channel mixing matrix (simulates domain shift)
    M = rng.randn(n_channels, n_channels)
    A = np.linalg.cholesky(M @ M.T + n_channels * np.eye(n_channels))
    X_list, y_list = [], []
    for cls in range(n_classes):
        scale = np.ones(n_channels)
        scale[cls] = 0.001  # class k: channel k has 1.5× higher std → survives lwf centering
        noise = rng.randn(n_trials_per_class, n_channels, n_times)
        noise *= scale[:, None]
        X_cls = np.einsum("ij,tjk->tik", A, noise)  # apply domain shift
        X_list.append(X_cls)
        y_list.append(np.full(n_trials_per_class, cls))
    return np.concatenate(X_list), np.concatenate(y_list)


X_per_subj = []
y_per_subj = []
for s in range(n_subjects):
    X_s, y_s = make_subject_data(
        n_trials_per_class, n_channels, n_times, n_classes, subj_seed=seed + s
    )
    X_per_subj.append(X_s)
    y_per_subj.append(y_s)

##############################################################################
# Pipelines
# ---------
#
# Shared preprocessing: raw EEG → Ledoit-Wolf covariance matrices.
# Non-TL pipelines apply preprocessing + classifier directly.
# TL pipelines wrap everything in an Adapter with Riemannian alignment
# (TLCenter → TLScale → TLRotate → TLClassifier), consistent with the
# full study in noplot_nch_study.py.

sf = make_pipeline(Covariances(estimator="lwf"))

pipelines = {}

pipelines["MDM"] = make_pipeline(sf, MDM())

pipelines["NCH"] = make_pipeline(
    sf,
    QuanticNCH(
        quantum=False,
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        subsampling="min",
    ),
)


def make_tl_pipeline(estimator):
    return Adapter(
        preprocessing=sf,
        estimator=make_pipeline(
            TLCenter(target_domain=None),
            TLScale(target_domain=None, centered_data=True),
            TLRotate(target_domain=None),
            TLClassifier(
                target_domain=None, estimator=estimator, domain_weight=None
            ),
        ),
    )


pipelines["MDM+TL"] = make_tl_pipeline(MDM())
pipelines["NCH+TL"] = make_tl_pipeline(
    QuanticNCH(
        quantum=False,
        seed=seed,
        n_samples_per_hull=n_samples_per_hull,
        subsampling="min",
    )
)

print("Total pipelines to evaluate:", len(pipelines))

##############################################################################
# Evaluation: Manual Leave-One-Subject-Out
# ----------------------------------------
#
# Mimics TLCrossSubjectEvaluation: for each test subject, train on all others
# and evaluate on the held-out subject. TL pipelines (Adapter) receive groups
# and target_domain; non-TL pipelines receive only X and y.

results = []
for test_subj in range(n_subjects):
    train_subjs = [s for s in range(n_subjects) if s != test_subj]
    X_train = np.concatenate([X_per_subj[s] for s in train_subjs])
    y_train = np.concatenate([y_per_subj[s] for s in train_subjs])
    groups_train = np.concatenate(
        [np.full(len(y_per_subj[s]), str(s)) for s in train_subjs]
    )
    X_test, y_test = X_per_subj[test_subj], y_per_subj[test_subj]

    for name, clf in pipelines.items():
        clf_ = deepcopy(clf)
        if "target_domain" in inspect.signature(clf_.fit).parameters:
            clf_.fit(
                X_train,
                y_train,
                groups=groups_train,
                target_domain=groups_train[0],
            )
        else:
            clf_.fit(X_train, y_train)
        score = roc_auc_score(y_test, clf_.predict_proba(X_test)[:, 1])
        results.append({"pipeline": name, "subject": test_subj, "score": score})
        print(f"  subject={test_subj}, pipeline={name}: auc={score:.3f}")

results = pd.DataFrame(results)
print("\nResults:")
print(results)
print("\nMean ROC AUC per pipeline:")
print(results.groupby("pipeline")[["score"]].mean())

##############################################################################
# Plot Results
# ------------
#
# Strip + point plot of per-subject accuracy for each pipeline.
# TL variants are shown first to visually compare the effect of alignment.

order = ["NCH+TL", "MDM+TL", "NCH", "MDM"]

fig, ax = plt.subplots(facecolor="white", figsize=[6, 4])

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
)
sns.pointplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    palette="Set1",
    order=order,
)

ax.set_ylabel("ROC AUC")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.legend(fontsize=8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##############################################################################
# 3D Manifold: Before and After RPA
# ----------------------------------
#
# A 2×2 SPD matrix has exactly 3 unique entries: [[a, b], [b, d]] → (a, b, d).
# Whitening(n_components=2) maps each 4×4 covariance to 2×2, giving exact 3D
# coordinates on the SPD cone via to_xyz — no approximation, no PCA.
# Color = subject, marker = class (o / ^).
#
# Before RPA: each subject's cloud sits in a different region of the cone
#             (domain shift visible as scattered subject clusters).
# After  RPA: subjects rotate toward the reference domain — clusters merge
#             and class structure becomes the dominant axis.

# Re-compute covariance matrices for all subjects
cov_pipe = Covariances(estimator="lwf")
covs_list, subj_arr, class_arr, domain_arr = [], [], [], []
for s in range(n_subjects):
    C_s = cov_pipe.fit_transform(X_per_subj[s])
    n_s = len(y_per_subj[s])
    covs_list.append(C_s)
    subj_arr.append(np.full(n_s, s))
    class_arr.append(y_per_subj[s])
    domain_arr.append(np.full(n_s, str(s), dtype=object))

covs_raw = np.concatenate(covs_list)
subj_arr = np.concatenate(subj_arr)
class_arr = np.concatenate(class_arr)
domain_arr = np.concatenate(domain_arr)

# Encode domains + classes for pyriemann TL transformers
_, y_enc = encode_domains(covs_raw, class_arr, domain_arr)

# Fit RPA: align all subjects toward subject "0" as reference
target_domain = "0"
covs_c = TLCenter(target_domain=target_domain).fit_transform(covs_raw, y_enc)
covs_s = TLScale(
    target_domain=target_domain, centered_data=True
).fit_transform(covs_c, y_enc)
covs_aligned = TLRotate(target_domain=target_domain).fit_transform(covs_s, y_enc)


def to_3d_coords(covs):
    """4×4 SPD → 2×2 via whitening → exact 3D coordinates (a, b, d)."""
    covs2x2 = Whitening(dim_red={"n_components": 2}).fit_transform(covs)
    return to_xyz(covs2x2)


coords_before = to_3d_coords(covs_raw)
coords_after = to_3d_coords(covs_aligned)

subject_colors = plt.cm.tab10(np.linspace(0, 1, n_subjects))
markers = ["o", "^"]  # class 0, class 1

fig = plt.figure(figsize=(14, 6), facecolor="white")
fig.suptitle("SPD manifold — whitening to 2×2 + xyz (3D)", fontsize=13)

for ax_idx, (coords, title) in enumerate(
    [(coords_before, "Before RPA"), (coords_after, "After RPA")]
):
    ax = fig.add_subplot(1, 2, ax_idx + 1, projection="3d")
    ax.set_title(title)
    for s in range(n_subjects):
        for cls in range(n_classes):
            mask = (subj_arr == s) & (class_arr == cls)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                color=subject_colors[s],
                marker=markers[cls],
                alpha=0.7,
                s=40,
            )
    ax.set_xlabel("σ₁²")
    ax.set_ylabel("σ₁₂")
    ax.set_zlabel("σ₂²")

legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=subject_colors[s], markersize=8, label=f"S{s}",
    )
    for s in range(n_subjects)
] + [
    plt.Line2D(
        [0], [0], marker=m, color="gray",
        markersize=8, linestyle="None", label=f"class {cls}",
    )
    for cls, m in enumerate(markers)
]
fig.legend(
    handles=legend_handles, loc="center right",
    fontsize=7, ncol=1, bbox_to_anchor=(1.02, 0.5),
)
plt.tight_layout()
plt.show()

##############################################################################
# Same 3D manifold, coloured by class instead of subject

class_colors = ["red", "blue"]

fig = plt.figure(figsize=(14, 6), facecolor="white")
fig.suptitle("SPD manifold — whitening to 2×2 + xyz (3D, colour = class)", fontsize=13)

for ax_idx, (coords, title) in enumerate(
    [(coords_before, "Before RPA"), (coords_after, "After RPA")]
):
    ax = fig.add_subplot(1, 2, ax_idx + 1, projection="3d")
    ax.set_title(title)
    for cls in range(n_classes):
        mask = class_arr == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            color=class_colors[cls],
            alpha=0.5,
            s=40,
            label=f"class {cls}",
        )
    ax.set_xlabel("σ₁²")
    ax.set_ylabel("σ₁₂")
    ax.set_zlabel("σ₂²")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
