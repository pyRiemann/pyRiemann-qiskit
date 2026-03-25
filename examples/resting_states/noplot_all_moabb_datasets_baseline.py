"""
====================================================================
Baseline classification of resting-state datasets from MOABB using CSP and MDM
====================================================================

Within-session evaluation of CSP+LDA and MDM pipelines across three
resting-state datasets: Hinss2021, Rodrigues2017, and Cattan2019_PHMD.
No transfer learning is applied.

"""
# Author: Gregoire Cattan
# Modified from noplot_all_moabb_datasets.py
# License: BSD (3-clause)

import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moabb import set_log_level
import moabb.analysis.plotting as moabb_plt
from moabb.datasets import Cattan2019_PHMD, Hinss2021, Rodrigues2017
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline

print(__doc__)

set_log_level("info")

##############################################################################
# Set global seed for better reproducibility

seed = round(time.time())
print(seed)

random.seed(seed)
np.random.seed(seed)

##############################################################################
# PSD per dataset (channel Cz)
# ----------------------------
#
# Visualise the mean power spectrum per class for each dataset using
# paradigm.psd() (Welch's method). No MOABB evaluation is involved.


def _find_channel(dataset, target="fp1"):
    """Return the ch_names entry that matches target case-insensitively."""
    subject = dataset.subject_list[0]
    sessions = dataset.get_data(subjects=[subject])
    raw = list(list(list(sessions.values())[0].values())[0].values())[0]
    for ch in raw.ch_names:
        if ch.lower() == target.lower():
            return ch
    raise ValueError(f"'{target}' not found. Available: {raw.ch_names}")


_cattan = Cattan2019_PHMD()
_rodrigues = Rodrigues2017()
_hinss = Hinss2021()

psd_configs = [
    (
        "Cattan2019-PHMD",
        _cattan,
        RestingStateToP300Adapter(
            events=["on", "off"], channels=[_find_channel(_cattan)]
        ),
        ["on", "off"],
        1,
    ),
    (
        "Rodrigues2017",
        _rodrigues,
        RestingStateToP300Adapter(
            fmin=1,
            fmax=35,
            events=["closed", "open"], channels=[_find_channel(_rodrigues)]
        ),
        ["closed", "open"],
        1,
    ),
    (
        "Hinss2021",
        _hinss,
        RestingStateToP300Adapter(
            events=["easy", "medium"],
            tmin=0,
            tmax=0.5,
            channels=[_find_channel(_hinss)],
        ),
        ["easy", "medium"],
        1,
    ),
]

fig, axes = plt.subplots(1, 3, facecolor="white", figsize=[14, 4])

for ax, (title, dataset, paradigm, events, subject) in zip(axes, psd_configs):
    f, S, _, y = paradigm.psd(subject, dataset)
    for event in events:
        mean_power = np.mean(S[y == event], axis=0).flatten()
        ax.plot(f, 10 * np.log10(mean_power), label=event)
    ax.set_xlim(paradigm.fmin, paradigm.fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()


##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

sf = make_pipeline(
    Covariances(estimator="lwf"),
)

pipelines = {}

pipelines["CSP+MDM"] = make_pipeline(
    sf,
    CSP(nfilter=8, log=False),
    MDM(),
)

print("Total pipelines to evaluate: ", len(pipelines))

overwrite = True  # set to True if we want to overwrite cached results

##############################################################################
# Evaluations
# -----------
#
# One WithinSession evaluation per dataset. Each dataset requires its own
# paradigm due to different event codes.

# --- Cattan2019_PHMD: WithinSession ---
paradigm_cattan = RestingStateToP300Adapter(events=dict(on=0, off=1))
evaluation_cattan = WithinSessionEvaluation(
    paradigm=paradigm_cattan,
    datasets=[Cattan2019_PHMD()],
    suffix="baseline_cattan",
    overwrite=overwrite,
    n_splits=3,
)
results_cattan = evaluation_cattan.process(pipelines)
print("Cattan pipelines computed:", results_cattan["pipeline"].unique().tolist())
print("Cattan results:")
print(results_cattan.groupby("pipeline")[["score", "time"]].mean())

# --- Rodrigues2017: WithinSession ---
paradigm_rodrigues = RestingStateToP300Adapter(events=dict(closed=2, open=1))
evaluation_rodrigues = WithinSessionEvaluation(
    paradigm=paradigm_rodrigues,
    datasets=[Rodrigues2017()],
    suffix="baseline_rodrigues",
    overwrite=overwrite,
    n_splits=3,
)
results_rodrigues = evaluation_rodrigues.process(pipelines)
results_rodrigues["score"] = 1 - results_rodrigues["score"]
print("Rodrigues results:")
print(results_rodrigues.groupby("pipeline")[["score", "time"]].mean())

# --- Hinss2021: WithinSession ---
paradigm_hinss = RestingStateToP300Adapter(
    events=dict(easy=2, medium=3), tmin=0, tmax=0.5
)
evaluation_hinss = WithinSessionEvaluation(
    paradigm=paradigm_hinss,
    datasets=[Hinss2021()],
    suffix="baseline_hinss",
    overwrite=overwrite,
    n_splits=3,
)
results_hinss = evaluation_hinss.process(pipelines)
print("Hinss results:")
print(results_hinss.groupby("pipeline")[["score", "time"]].mean())

##############################################################################
# Aggregate Results
# -----------------

results = pd.concat(
    [results_cattan, results_rodrigues, results_hinss],
    ignore_index=True,
)

print(results)
print("Averaging the session performance:")
print(results.groupby("pipeline")[["score", "time"]].mean())

##############################################################################
# Plot Results: Per-Dataset Comparison
# -------------------------------------

active_pipelines = results["pipeline"].unique().tolist()
dodge = len(active_pipelines) > 1

fig2, ax2 = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="dataset",
    hue="pipeline",
    ax=ax2,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
    hue_order=active_pipelines,
    dodge=dodge,
)
sns.pointplot(
    data=results,
    y="score",
    x="dataset",
    hue="pipeline",
    ax=ax2,
    palette="Set1",
    hue_order=active_pipelines,
    dodge=dodge,
)

ax2.set_ylabel("ROC AUC")
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[: len(active_pipelines)], labels[: len(active_pipelines)], title="Pipeline")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

