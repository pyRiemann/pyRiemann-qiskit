"""
====================================================================
Intrinsic dimension of IGL on the SPD manifold across EEG paradigms
====================================================================

Ablation study measuring the **effective intrinsic dimension** discovered by
``IGLSklearnClassifier`` (Hard Concrete gates) when operating on Riemannian
tangent-space features derived from EEG covariance matrices.

Five EEG paradigms are compared, each with three representative datasets:

- **P300**: bi2012, Cattan2019_VR, BNCI2014-009 — time-locked visual evoked
  potential.  Uses ``ERPCovariances`` to augment the covariance with the ERP
  template.
- **Motor Imagery**: AlexMI, BNCI2014-001, PhysionetMI — mu/beta
  event-related desynchronization during imagined movement.
  Uses ``Covariances(lwf)``.
- **Resting State**: Cattan2019_PHMD, Rodrigues2017, Hinss2021 — relaxed
  rest between distinct mental states.  Uses ``Covariances(lwf)``.
- **SSVEP**: Lee2019_SSVEP, Wang2016, MAMEM3 — steady-state visual evoked
  potential driven by flickering stimuli.  Uses ``Covariances(lwf)``.
- **CVEP**: Thielen2021, Thielen2015, CastillosCVEP40 — code-modulated
  visual evoked potential.  Uses ``Covariances(lwf)``.

Evaluation uses MOABB ``CrossSubjectEvaluation`` (train on N-1 subjects,
test on 1), which tests cross-subject generalization rather than
within-subject performance.

Preprocessing is fitted **inside each fold** by MOABB to avoid data leakage
(important for ``ERPCovariances``, which uses labels to compute the template).

A **TS+LR baseline** (TangentSpace + LogisticRegression) is evaluated
alongside IGL to provide a reference AUC for each dataset.

The ablation factors are:

- **operator**: gaussian, cauchy, laplacian, mexican_hat
- **max_dim**: 64 (fixed upper bound; d_eff <= max_dim is the measured quantity)

The key metric is **effective dimension** (d_eff): the number of Hard
Concrete gates with activation probability > 0.5.
"""
# Author: Gregoire Cattan
# License: BSD (3-clause)

import shutil
import warnings
from collections import defaultdict

# shutil.copy_tree was removed in Python 3.12; MOABB still uses the old name
if not hasattr(shutil, "copy_tree"):
    shutil.copy_tree = shutil.copytree

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moabb.datasets import (
    BNCI2014_001,
    BNCI2014_009,
    MAMEM3,
    AlexMI,
    CastillosCVEP40,
    Cattan2019_PHMD,
    Cattan2019_VR,
    Hinss2021,
    Lee2019_SSVEP,
    PhysionetMI,
    Rodrigues2017,
    Thielen2015,
    Thielen2021,
    Wang2016,
    bi2012,
)
from moabb.paradigms import CVEP, P300, SSVEP, MotorImagery, RestingStateToP300Adapter
from pyriemann.estimation import Covariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, TLClassifier, TLScale
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification.igl_reference import IGLSklearnClassifier, VPConfig
from pyriemann_qiskit.utils.transfer import Adapter, TLCrossSubjectEvaluation

warnings.filterwarnings("ignore")

print(__doc__)

##############################################################################
# Parameters

seed = 42
n_splits = 3
n_subjects = 5  # subjects per dataset (keep runtime reasonable)

# Ablation grid
operators = ["gaussian", "cauchy", "laplacian", "mexican_hat"]
max_dim = 64  # upper bound; d_eff <= max_dim is the actual measured quantity

# Fixed IGL hyper-params (not ablated)
n_anchors = 64
n_scales = 4
epochs = 1000
warmup_epochs = 200

##############################################################################
# Eff-dim capture
# ---------------
#
# MOABB evaluation does not expose fitted estimators, so we capture
# effective_dimension() via module-level per-operator logs appended inside
# each wrapper's fit().  Logs are cleared before every dataset evaluation and
# read immediately after, so they contain only entries for that dataset in
# the same (subject → session → fold) order as the MOABB results DataFrame.

_eff_dim_logs = defaultdict(list)
_tangent_dim_logs = defaultdict(list)


class _IGLGaussian(IGLSklearnClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        _eff_dim_logs["gaussian"].append(self.effective_dimension())
        _tangent_dim_logs["gaussian"].append(X.shape[1])
        return self


class _IGLCauchy(IGLSklearnClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        _eff_dim_logs["cauchy"].append(self.effective_dimension())
        _tangent_dim_logs["cauchy"].append(X.shape[1])
        return self


class _IGLLaplacian(IGLSklearnClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        _eff_dim_logs["laplacian"].append(self.effective_dimension())
        _tangent_dim_logs["laplacian"].append(X.shape[1])
        return self


class _IGLMexicanHat(IGLSklearnClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        _eff_dim_logs["mexican_hat"].append(self.effective_dimension())
        _tangent_dim_logs["mexican_hat"].append(X.shape[1])
        return self


_IGL_CLASSES = {
    "gaussian": _IGLGaussian,
    "cauchy": _IGLCauchy,
    "laplacian": _IGLLaplacian,
    "mexican_hat": _IGLMexicanHat,
}

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
paradigm_ssvep = SSVEP(n_classes=2, resample=128)
paradigm_cvep = CVEP(resample=128)

cov_erp = ERPCovariances(estimator="oas")
cov_lwf = Covariances(estimator="lwf")


def make_dataset(cls, n):
    d = cls()
    d.subject_list = d.subject_list[:n]
    return d


PARADIGMS = [
    # P300 (3)
    ("P300",         "P300 (bi2012)",        paradigm_p300,         cov_erp, make_dataset(bi2012, n_subjects)),
    ("P300",         "P300 (Cattan2019-VR)", paradigm_p300,         cov_erp, make_dataset(Cattan2019_VR, n_subjects)),
    ("P300",         "P300 (BNCI2014-009)",  paradigm_p300,         cov_erp, make_dataset(BNCI2014_009, n_subjects)),
    # Motor Imagery (3)
    ("MotorImagery", "MI (AlexMI)",          paradigm_mi,           cov_lwf, make_dataset(AlexMI, n_subjects)),
    ("MotorImagery", "MI (BNCI2014-001)",    paradigm_mi,           cov_lwf, make_dataset(BNCI2014_001, n_subjects)),
    ("MotorImagery", "MI (PhysionetMI)",     paradigm_mi,           cov_lwf, make_dataset(PhysionetMI, n_subjects)),
    # Resting State (3) — each uses a different paradigm object
    ("RestingState", "RS (PHMD)",            paradigm_rs_cattan,    cov_lwf, make_dataset(Cattan2019_PHMD, n_subjects)),
    ("RestingState", "RS (Rodrigues2017)",   paradigm_rs_rodrigues, cov_lwf, make_dataset(Rodrigues2017, n_subjects)),
    ("RestingState", "RS (Hinss2021)",       paradigm_rs_hinss,     cov_lwf, make_dataset(Hinss2021, n_subjects)),
    # SSVEP (3)
    ("SSVEP",        "SSVEP (Lee2019)",      paradigm_ssvep,        cov_lwf, make_dataset(Lee2019_SSVEP, n_subjects)),
    ("SSVEP",        "SSVEP (Wang2016)",     paradigm_ssvep,        cov_lwf, make_dataset(Wang2016, n_subjects)),
    ("SSVEP",        "SSVEP (MAMEM3)",       paradigm_ssvep,        cov_lwf, make_dataset(MAMEM3, n_subjects)),
    # CVEP (3)
    # ("CVEP",         "CVEP (Thielen2021)",   paradigm_cvep,         cov_lwf, make_dataset(Thielen2021, n_subjects)),
    #("CVEP",         "CVEP (Thielen2015)",   paradigm_cvep,         cov_lwf, make_dataset(Thielen2015, n_subjects)),
    #("CVEP",         "CVEP (Castillos40)",   paradigm_cvep,         cov_lwf, make_dataset(CastillosCVEP40, n_subjects)),
]

##############################################################################
# Pipeline factory
# ----------------
#
# Returns one TS+LR pipeline and one IGL pipeline per operator.
# cov_est is cloned per pipeline so each has an independent estimator state.


def _rpa_head():
    """Return (TLCenter, TLScale) — the two RPA alignment steps."""
    return (
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
    )


def make_pipelines(cov_est):
    pipes = {
        "TS+LR": Adapter(
            preprocessing=make_pipeline(clone(cov_est)),
            estimator=make_pipeline(
                *_rpa_head(),
                TLClassifier(
                    target_domain=None,
                    estimator=make_pipeline(
                        TangentSpace(metric="riemann"),
                        LogisticRegression(max_iter=1000),
                    ),
                    domain_weight=None,
                ),
            ),
        ),
    }
    for op in operators:
        cls = _IGL_CLASSES[op]
        pipes[f"IGL-{op}"] = Adapter(
            preprocessing=make_pipeline(clone(cov_est)),
            estimator=make_pipeline(
                *_rpa_head(),
                TLClassifier(
                    target_domain=None,
                    estimator=make_pipeline(
                        TangentSpace(metric="riemann"),
                        cls(
                            max_dim=max_dim,
                            n_anchors=n_anchors,
                            n_scales=n_scales,
                            operator=op,
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
                            random_state=seed,
                        ),
                    ),
                    domain_weight=None,
                ),
            ),
        )
    return pipes


##############################################################################
# MOABB evaluation loop
# ----------------------
#
# One CrossSubjectEvaluation per dataset (necessary because each dataset may
# use a different paradigm object, especially the three RS datasets).
# MOABB handles train/test splitting (GroupKFold with subjects as groups),
# pipeline cloning, and AUC scoring entirely within process().

all_results = []

for paradigm_type, display_name, paradigm, cov_est, dataset in PARADIGMS:
    print(f"\n{'='*60}")
    print(f"Dataset: {display_name}")
    print(f"{'='*60}")

    # Clear logs so they contain only entries for this dataset
    for op in operators:
        _eff_dim_logs[op].clear()
        _tangent_dim_logs[op].clear()

    pipelines = make_pipelines(cov_est)

    try:
        evaluation = TLCrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            suffix="igl_ablation",
            overwrite=True,
            n_splits=n_splits,
            random_state=seed,
        )
        res = evaluation.process(pipelines)
    except Exception as exc:
        print(f"  [SKIP] Evaluation failed: {exc}")
        continue

    # Tag with paradigm metadata
    res["paradigm_type"] = paradigm_type
    res["display_name"] = display_name

    # Initialize columns so they exist even when results come from cache
    res["eff_dim"] = float("nan")
    res["tangent_dim"] = float("nan")

    # Attach eff_dim / tangent_dim to IGL rows.
    # MOABB processes subjects → sessions → folds sequentially, so the log
    # is in the same order as the IGL rows in the results DataFrame.
    for op in operators:
        mask = res["pipeline"] == f"IGL-{op}"
        log = _eff_dim_logs[op]
        tdim = _tangent_dim_logs[op]
        n_rows = int(mask.sum())
        if len(log) == n_rows:
            res.loc[mask, "eff_dim"] = log
            res.loc[mask, "tangent_dim"] = tdim
        else:
            print(
                f"  [WARN] eff_dim log mismatch for IGL-{op}: "
                f"log={len(log)}, rows={n_rows}"
            )

    all_results.append(res)
    print(res.groupby("pipeline")[["score"]].mean().round(3).to_string())

##############################################################################
# Tidy the results DataFrame

df = pd.concat(all_results, ignore_index=True)
df.drop(columns=["dataset"], inplace=True, errors="ignore")
df.rename(columns={"score": "auc", "display_name": "dataset"}, inplace=True)

df["model"] = df["pipeline"].apply(
    lambda p: "IGL" if p.startswith("IGL-") else p
)
df["operator"] = df["pipeline"].apply(
    lambda p: p[len("IGL-"):] if p.startswith("IGL-") else None
)
df["compression_ratio"] = df["eff_dim"] / df["tangent_dim"]

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

PALETTE = {
    "P300":         "tab:blue",
    "MotorImagery": "tab:orange",
    "RestingState": "tab:green",
    "SSVEP":        "tab:purple",
    "CVEP":         "tab:red",
}

fig, ax = plt.subplots(facecolor="white", figsize=(7, 4))
sns.boxplot(data=df_igl, x="paradigm_type", y="eff_dim", palette=PALETTE, ax=ax)
sns.stripplot(
    data=df_igl, x="paradigm_type", y="eff_dim",
    hue="dataset", dodge=False, alpha=0.5, jitter=True, ax=ax,
)
ax.set_ylabel("Effective dimension $d_{\\mathrm{eff}}$")
ax.set_xlabel("Paradigm")
ax.set_title("Intrinsic dimension discovered by IGL per EEG paradigm (5 paradigms × 3 datasets)")
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

fig, ax = plt.subplots(facecolor="white", figsize=(16, 4))
dataset_order = [d for _, d, *_ in PARADIGMS]
sns.boxplot(
    data=df_igl, x="dataset", y="eff_dim",
    hue="paradigm_type", order=dataset_order,
    hue_order=["P300", "MotorImagery", "RestingState", "SSVEP", "CVEP"],
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
    sharey=True, col_order=["P300", "MotorImagery", "RestingState", "SSVEP", "CVEP"],
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
