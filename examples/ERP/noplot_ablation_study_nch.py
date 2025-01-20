"""
====================================================================
Ablation study for the NCH
====================================================================

This example is an ablation study of the NCH.
Two subsampling strategies (min and random) are benchmarked,
varying the number of hull and samples.

We used the dataset bi2012 for this study.

"""
# Author: Gregoire Cattan, Quentin Barthelemy
# License: BSD (3-clause)

import random
import warnings

import numpy as np
import qiskit_algorithms
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticNCH

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

events = ["on", "off"]
paradigm = P300()

datasets = [bi2012()]

for dataset in datasets:
    dataset.subject_list = dataset.subject_list[0:-1]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

##############################################################################
# Set seed

seed = 475751

random.seed(seed)
np.random.seed(seed)
qiskit_algorithms.utils.algorithm_globals.random_seed


##############################################################################
# Set NCH strategy

strategy = "random"  # or "random"
max_hull_per_class = 1 if strategy == "min" else 6
max_samples_per_hull = 15 if strategy == "min" else 25
samples_step = 1 if strategy == "min" else 5

##############################################################################
# Define spatial filtering

sf = make_pipeline(XdawnCovariances())

##############################################################################
# Define pipelines

for n_hulls_per_class in range(1, max_hull_per_class + 1, 1):
    for n_samples_per_hull in range(1, max_samples_per_hull + 1, samples_step):
        pipe_name = strategy.upper()
        key = f"NCH+{pipe_name}_HULL_{n_hulls_per_class}h_{n_samples_per_hull}samples"
        print(key)
        pipelines[key] = make_pipeline(
            sf,
            QuanticNCH(
                seed=seed,
                n_hulls_per_class=n_hulls_per_class,
                n_samples_per_hull=n_samples_per_hull,
                n_jobs=12,
                subsampling=strategy,
                quantum=False,
            ),
        )


print("Total pipelines to evaluate: ", len(pipelines))
print(np.unique(pipelines.keys()))

##############################################################################
# Run evaluation

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    suffix="examples",
    overwrite=overwrite,
    n_splits=None,
    random_state=seed,
)


results = evaluation.process(pipelines)

##############################################################################
# Print results


def get_hull(v):
    return int(v.split("NCH+MIN_HULL_")[1].split("h_")[0])


def get_samples(v):
    return int(v.split("h_")[1].split("samples")[0])


results["n_hull"] = results["pipeline"].apply(get_hull)
results["n_samples"] = results["pipeline"].apply(get_samples)
print(results)

means = results.groupby("pipeline").mean()

if strategy == "random":
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(
        means.n_hull, means.n_samples, means.score, cmap=cm.jet, linewidth=0.2
    )
    ax.set_xlabel("n_hull")
    ax.set_ylabel("n_samples")
    ax.set_zlabel("score")
else:
    sns.pointplot(means, x="n_samples", y="score")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

plt.show()
