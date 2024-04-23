"""
====================================================================
Light Benchmark
====================================================================

This benchmark is a non-regression performance test, intended
to run on Ci with each PRs.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_P300_bi.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.paradigms import P300
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
    QuantumMDMWithRiemannianPipeline,
)
from pyriemann_qiskit.classification import QuanticNCH
import warnings
import sys

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Prepare data
# -------------
#
##############################################################################

paradigm = P300(resample=128)

dataset = bi2012()  # MOABB provides several other P300 datasets

X, y, _ = paradigm.get_data(dataset, subjects=[1])

# Reduce the dataset size for Ci
_, X, _, y = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)

y = LabelEncoder().fit_transform(y)

# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
##############################################################################

pipelines = {}

pipelines["RG_QSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100, nfilter=2, dim_red=PCA(n_components=5), params={"seed": 42}
)

pipelines["RG_VQC"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100, spsa_trials=1, two_local_reps=2, params={"seed": 42}
)

pipelines["QMDM_mean"] = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "qeuclid", "distance": "euclid"},
    quantum=True,
    regularization=Shrinkage(shrinkage=0.9),
)

pipelines["QMDM_dist"] = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "logeuclid", "distance": "qlogeuclid_hull"}, quantum=True
)

pipelines["RG_LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    PCA(n_components=5),
    LDA(solver="lsqr", shrinkage="auto"),
)

pipelines["NCH_MIN_HULL"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        # classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    QuanticNCH(
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=False,
    ),
)

##############################################################################
# Compute score
# --------------
#
##############################################################################

scores = {}

for key, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores[key] = score

print("Scores: ", scores)

##############################################################################
# Compare scores between PR and main branches
# -------------------------------------------
#
##############################################################################


def set_output(key: str, value: str):
    print(f"::set-output name={key}::{value}")  # noqa: E231


is_pr = sys.argv[1] == "pr"

if is_pr:
    for key, score in scores.items():
        set_output(key, score)
else:
    success = True
    i = 0
    for key, score in scores.items():
        i = i + 1
        pr_score = sys.argv[i]
        success = success and (True if float(pr_score) >= score else False)
    set_output("success", "1" if success else "0")
