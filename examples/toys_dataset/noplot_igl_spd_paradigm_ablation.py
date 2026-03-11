"""
====================================================================
Intrinsic dimension of IGL on the SPD manifold across EEG paradigms
====================================================================

Ablation study measuring the **effective intrinsic dimension** discovered by
``IGLSklearnClassifier`` (Hard Concrete gates) when operating on Riemannian
tangent-space features derived from EEG covariance matrices.

Three EEG paradigms are compared, each with multiple representative datasets:

- **P300**: bi2012, Cattan2019_VR — time-locked visual evoked potential.
  Uses ``ERPCovariances`` to augment the covariance with the ERP template.
- **Motor Imagery**: AlexMI, BNCI2014-001 — mu/beta event-related
  desynchronization during imagined movement.  Uses ``Covariances(lwf)``.
- **Resting State**: Cattan2019_PHMD, Rodrigues2017, Hinss2021 — relaxed
  rest between distinct mental states.  Uses ``Covariances(lwf)``.

Preprocessing is fitted **inside each fold** to avoid data leakage
(important for ``ERPCovariances``, which uses labels to compute the template).

A **TS+LR baseline** (TangentSpace + LogisticRegression) is evaluated alongside IGL in each fold to provide
a reference AUC for each dataset.

The ablation factors are:

- **operator**: gaussian, cauchy, laplacian, mexican_hat
- **max_dim**: 32 (fixed upper bound; d_eff <= max_dim is the measured quantity)

The key metric is **effective dimension** (d_eff): the number of Hard
Concrete gates with activation probability > 0.5.
"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import shutil
import warnings

# shutil.copy_tree was removed in Python 3.12; MOABB still uses the old name
if not hasattr(shutil, "copy_tree"):
    shutil.copy_tree = shutil.copytree

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moabb.datasets import (
    AlexMI,
    BNCI2014_001,
    Cattan2019_PHMD,
    Cattan2019_VR,
    Hinss2021,
    Rodrigues2017,
    bi2012,
)
from moabb.paradigms import MotorImagery, P300, RestingStateToP300Adapter
from pyriemann.estimation import Covariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from pyriemann_qiskit.classification.igl_reference import (
    IGLSklearnClassifier,
    VPConfig,
)

warnings.filterwarnings("ignore")

print(__doc__)

##############################################################################
# Parameters

seed = 42
n_splits = 3
n_subjects = 5  # subjects per dataset (keep runtime reasonable)

# Ablation grid
operators = ["gaussian", "cauchy", "laplacian", "mexican_hat"]
max_dim = 32  # upper bound; d_eff <= max_dim is the actual measured quantity

# Fixed IGL hyper-params (not ablated)
n_anchors = 64
n_scales = 4
epochs = 1000
warmup_epochs = 200

##############################################################################
# Dataset and paradigm definitions
# ---------------------------------
#
# Each entry: (paradigm_type, display_name, paradigm_obj, covariance_estimator,
#              dataset).
# P300 uses ERPCovariances; MI and RS use plain Covariances(lwf).

paradigm_p300 = P300(resample=128)
paradigm_mi = MotorImagery(events=["right_hand", "feet"], n_classes=2, resample=128)
paradigm_rs_cattan = RestingStateToP300Adapter(events=dict(on=0, off=1))
paradigm_rs_rodrigues = RestingStateToP300Adapter(events=dict(closed=1, open=2))
paradigm_rs_hinss = RestingStateToP300Adapter(
    events=dict(easy=2, medium=3), tmin=0, tmax=0.5
)

cov_erp = ERPCovariances(estimator="oas")
cov_lwf = Covariances(estimator="lwf")


def make_dataset(cls, n):
    d = cls()
    d.subject_list = d.subject_list[:n]
    return d


PARADIGMS = [
    ("P300",         "P300 (bi2012)",        paradigm_p300,         cov_erp, make_dataset(bi2012, n_subjects)),
    ("P300",         "P300 (Cattan2019-VR)", paradigm_p300,         cov_erp, make_dataset(Cattan2019_VR, n_subjects)),
    ("MotorImagery", "MI (AlexMI)",          paradigm_mi,           cov_lwf, make_dataset(AlexMI, n_subjects)),
    ("MotorImagery", "MI (BNCI2014-001)",    paradigm_mi,           cov_lwf, make_dataset(BNCI2014_001, n_subjects)),
    ("RestingState", "RS (PHMD)",            paradigm_rs_cattan,    cov_lwf, make_dataset(Cattan2019_PHMD, n_subjects)),
    ("RestingState", "RS (Rodrigues2017)",   paradigm_rs_rodrigues, cov_lwf, make_dataset(Rodrigues2017, n_subjects)),
    ("RestingState", "RS (Hinss2021)",       paradigm_rs_hinss,     cov_lwf, make_dataset(Hinss2021, n_subjects)),
]

##############################################################################
# Cross-validation loop
# ----------------------
#
# Preprocessing is fitted on the training fold only (no data leakage).
# TS+LR is evaluated in the same fold as a reference baseline.

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
le = LabelEncoder()

records = []

for paradigm_type, display_name, paradigm, cov_est, dataset in PARADIGMS:
    print(f"\n{'='*60}")
    print(f"Dataset: {display_name}")
    print(f"{'='*60}")

    try:
        X_raw, y_str, metadata = paradigm.get_data(dataset)
    except Exception as exc:
        print(f"  [SKIP] Could not load data: {exc}")
        continue

    y = le.fit_transform(y_str)
    n_channels = X_raw.shape[1]
    print(f"  Loaded: {X_raw.shape[0]} trials, {n_channels} channels")
    tangent_dim = None  # resolved after first fold preprocessing

    n_total_configs = len(operators)
    cfg_idx = 0

    # TS+LR baseline: one AUC per fold, shared across all operators
    lr_aucs = []
    for train_idx, test_idx in cv.split(X_raw, y):
        X_raw_tr, X_raw_te = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            lr_pipe = make_pipeline(cov_est, TangentSpace(metric="riemann"), LogisticRegression(max_iter=1000))
            lr_pipe.fit(X_raw_tr, y_tr)
            lr_proba = lr_pipe.predict_proba(X_raw_te)
            lr_aucs.append(roc_auc_score(y_te, lr_proba[:, 1]))
        except Exception as exc:
            print(f"  [WARN] TS+LR fold failed: {exc}")
            lr_aucs.append(float("nan"))

    mean_lr = float(np.nanmean(lr_aucs))
    print(f"  TS+LR baseline: mean_auc={mean_lr:.3f}  folds={[f'{a:.3f}' for a in lr_aucs]}")

    for fold_i, auc_lda in enumerate(lr_aucs):
        records.append({
            "paradigm_type": paradigm_type,
            "dataset": display_name,
            "model": "TS+LR",
            "operator": None,
            "fold": fold_i,
            "auc": auc_lda,
            "eff_dim": float("nan"),
        })

    for operator in operators:
        cfg_idx += 1
        aucs, eff_dims = [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw, y)):
            X_raw_tr, X_raw_te = X_raw[train_idx], X_raw[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Fit covariances + tangent space on training data only
            try:
                ts = TangentSpace(metric="riemann")
                covs_tr = cov_est.fit_transform(X_raw_tr, y_tr)
                X_tr = ts.fit_transform(covs_tr, y_tr)
                covs_te = cov_est.transform(X_raw_te)
                X_te = ts.transform(covs_te)
                if tangent_dim is None:
                    tangent_dim = X_tr.shape[1]
                    print(f"  Tangent-space dim = {tangent_dim}")
            except Exception as exc:
                print(f"  [WARN] Preprocessing fold {fold} failed: {exc}")
                aucs.append(float("nan"))
                eff_dims.append(-1)
                continue

            clf = IGLSklearnClassifier(
                max_dim=max_dim,
                n_anchors=n_anchors,
                n_scales=n_scales,
                operator=operator,
                hidden=128,
                use_gates=True,
                encoder="mlp",
                training="vp",
                vp_config=VPConfig(
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    source_l2=0.01,
                    log_every=epochs,
                    verbose=False,
                ),
                random_state=seed + fold,
            )

            try:
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)
                auc = roc_auc_score(y_te, proba[:, 1])
                d_eff = clf.effective_dimension()
            except Exception as exc:
                print(f"    [WARN] IGL fold {fold} failed: {exc}")
                auc = float("nan")
                d_eff = -1

            aucs.append(auc)
            eff_dims.append(d_eff)

        mean_auc = float(np.nanmean(aucs))
        mean_eff = float(np.nanmean([d for d in eff_dims if d >= 0]))

        print(
            f"  [{cfg_idx:2d}/{n_total_configs}]  "
            f"op={operator:<12s}  "
            f"mean_auc={mean_auc:.3f}  "
            f"eff_dim={mean_eff:.1f}  "
            f"folds={[f'{a:.3f}' for a in aucs]}"
        )

        for fold_i, (auc_i, d_i) in enumerate(zip(aucs, eff_dims)):
            records.append({
                "paradigm_type": paradigm_type,
                "dataset": display_name,
                "model": "IGL",
                "operator": operator,
                "fold": fold_i,
                "auc": auc_i,
                "eff_dim": d_i,
                "tangent_dim": tangent_dim,
                "compression_ratio": (
                    d_i / tangent_dim if (d_i >= 0 and tangent_dim) else float("nan")
                ),
            })

df = pd.DataFrame(records)
df_igl = df[df["model"] == "IGL"].copy()
df_lr = df[df["model"] == "TS+LR"].copy()

##############################################################################
# Summary table
# -------------

print("\n=== IGL summary (mean over folds and operators) ===")
print(
    df_igl.groupby(["paradigm_type", "dataset"])[["auc", "eff_dim"]]
    .mean()
    .round(3)
    .to_string()
)

print("\n=== TS+LR baseline (mean over folds) ===")
print(
    df_lr.groupby(["paradigm_type", "dataset"])[["auc"]]
    .mean()
    .round(3)
    .to_string()
)

##############################################################################
# Plot 1: Effective dimension by paradigm type
# ---------------------------------------------

PALETTE = {"P300": "tab:blue", "MotorImagery": "tab:orange", "RestingState": "tab:green"}

fig, ax = plt.subplots(facecolor="white", figsize=(7, 4))
sns.boxplot(data=df_igl, x="paradigm_type", y="eff_dim", palette=PALETTE, ax=ax)
sns.stripplot(
    data=df_igl, x="paradigm_type", y="eff_dim",
    hue="dataset", dodge=False, alpha=0.5, jitter=True, ax=ax,
)
ax.set_ylabel("Effective dimension $d_{\\mathrm{eff}}$")
ax.set_xlabel("Paradigm")
ax.set_title("Intrinsic dimension discovered by IGL per EEG paradigm")
ax.legend(title="Dataset", fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("intrinsec_dim_igl.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Plot 2: IGL AUC vs TS+LR AUC per dataset
# ------------------------------------------
#
# Each point is one dataset; error bars = std over folds and operators.

agg_igl = df_igl.groupby("dataset")["auc"].agg(["mean", "std"]).reset_index()
agg_lda = df_lr.groupby("dataset")["auc"].mean().reset_index().rename(columns={"auc": "lr_auc"})
agg = agg_igl.merge(agg_lda, on="dataset")
ptype_map = {d: t for t, d, *_ in PARADIGMS}
agg["paradigm_type"] = agg["dataset"].map(ptype_map)

fig, ax = plt.subplots(facecolor="white", figsize=(6, 5))
for ptype, grp in agg.groupby("paradigm_type"):
    ax.errorbar(
        grp["lr_auc"], grp["mean"],
        yerr=grp["std"], fmt="o", label=ptype,
        color=PALETTE.get(ptype), capsize=4,
    )
    for _, row in grp.iterrows():
        ax.annotate(
            row["dataset"].split("(")[-1].rstrip(")"),
            (row["lr_auc"], row["mean"]),
            fontsize=6, ha="left", va="bottom",
        )
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1]),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8, label="IGL = TS+LR")
ax.set_xlabel("TS+LR AUC (baseline)")
ax.set_ylabel("IGL AUC (mean ± std over operators/folds)")
ax.set_title("IGL vs TS+LR per dataset")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("intrinsec_igl_vs_tslda.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Plot 3: d_eff vs AUC (scatter, coloured by paradigm type)
# ----------------------------------------------------------

fig, ax = plt.subplots(facecolor="white", figsize=(6, 4))
for ptype, grp in df_igl.groupby("paradigm_type"):
    ax.scatter(grp["eff_dim"], grp["auc"], label=ptype,
               alpha=0.5, color=PALETTE.get(ptype))
ax.axhline(0.5, ls="--", color="grey", lw=0.8, label="Chance")
ax.set_xlabel("Effective dimension $d_{\\mathrm{eff}}$")
ax.set_ylabel("ROC AUC")
ax.set_title("AUC vs effective dimension per paradigm")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("intrinsec_auc_vs_dim.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Plot 4: d_eff per dataset
# -------------------------

fig, ax = plt.subplots(facecolor="white", figsize=(10, 4))
dataset_order = [d for _, d, *_ in PARADIGMS]
sns.boxplot(
    data=df_igl, x="dataset", y="eff_dim",
    hue="paradigm_type", order=dataset_order,
    hue_order=["P300", "MotorImagery", "RestingState"],
    palette=PALETTE, ax=ax, dodge=False,
)
ax.set_ylabel("Effective dimension $d_{\\mathrm{eff}}$")
ax.set_xlabel("")
ax.set_title("Effective dimension per dataset")
ax.legend(title="Paradigm", fontsize=8)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("intrinsec_dim_igl_per_dataset.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Plot 5: d_eff / tangent_dim per dataset and paradigm
# -----------------------------------------------------
#
# Compression ratio = d_eff / D where D = tangent-space dimension.
# Shows what fraction of the ambient space IGL actually uses.
# A low ratio means the manifold is intrinsically low-dimensional
# relative to the number of covariance entries.

fig, axes = plt.subplots(1, 2, facecolor="white", figsize=(12, 4))

# Left: boxplot per paradigm type (aggregated across datasets)
sns.boxplot(
    data=df_igl, x="paradigm_type", y="compression_ratio",
    palette=PALETTE, ax=axes[0],
)
sns.stripplot(
    data=df_igl, x="paradigm_type", y="compression_ratio",
    hue="dataset", dodge=False, alpha=0.4, jitter=True, ax=axes[0],
)
axes[0].set_ylabel("$d_{\\mathrm{eff}}$ / $D$")
axes[0].set_xlabel("Paradigm")
axes[0].set_title("Compression ratio by paradigm")
axes[0].legend(title="Dataset", fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")

# Right: per-dataset strip + mean, ordered as in PARADIGMS
agg_ratio = (
    df_igl.groupby("dataset")["compression_ratio"]
    .agg(["mean", "std"])
    .reindex(dataset_order)
    .reset_index()
)
ptype_colors = [PALETTE[ptype_map[d]] for d in dataset_order if d in ptype_map]
axes[1].barh(
    agg_ratio["dataset"], agg_ratio["mean"],
    xerr=agg_ratio["std"], color=ptype_colors, alpha=0.8, capsize=3,
)
axes[1].set_xlabel("$d_{\\mathrm{eff}}$ / $D$")
axes[1].set_ylabel("")
axes[1].set_title("Mean compression ratio per dataset")
axes[1].axvline(1.0, ls="--", color="grey", lw=0.8)

plt.tight_layout()
plt.savefig("intrinsec_compress_ratio_per_paradigm.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# Plot 6: d_eff by operator, faceted by paradigm type
# ----------------------------------------------------
#
# Shows how kernel choice affects the discovered intrinsic dimension,
# separately for each EEG paradigm type.

g = sns.FacetGrid(
    df_igl, col="paradigm_type", height=4, aspect=0.9,
    sharey=True, col_order=["P300", "MotorImagery", "RestingState"],
)
g.map_dataframe(
    sns.boxplot, x="operator", y="eff_dim",
    order=operators, palette="Set2",
)
g.map_dataframe(
    sns.stripplot, x="operator", y="eff_dim",
    order=operators, color="black", alpha=0.4, jitter=True, size=3,
)
g.set_axis_labels("Kernel operator", "Effective dimension $d_{\\mathrm{eff}}$")
g.set_titles(col_template="{col_name}")
g.figure.suptitle(
    "Intrinsic dimension by kernel operator and EEG paradigm", y=1.02
)
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("intrinsec_dim_igl_operator.png", dpi=150, bbox_inches="tight")
plt.show()
