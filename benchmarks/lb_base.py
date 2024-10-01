"""
====================================================================
Light Benchmark
====================================================================

Common script to run light benchmarks

"""
# Author: Gregoire Cattan
# Modified from plot_classify_P300_bi.py of pyRiemann
# License: BSD (3-clause)

import sys
import warnings

from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.paradigms import P300
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def _set_output(key: str, value: str):
    print(f"::set-output name={key}::{value}")  # noqa: E231


def run(pipelines):
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

    # Compute scores
    scores = {}

    for key, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)
        scores[key] = score

    print("Scores: ", scores)

    # Compare scores between PR and main branches
    is_pr = sys.argv[1] == "pr"

    if is_pr:
        for key, score in scores.items():
            _set_output(key, score)
    else:
        success = True
        i = 0
        for key, score in scores.items():
            i = i + 1
            pr_score = sys.argv[i]
            pr_score_trun = int(float(pr_score) * 100)
            score_trun = int(score * 100)
            better_pr_score = pr_score_trun >= score_trun
            success = success and better_pr_score
            print(
                f"{key}: {pr_score_trun} (PR) >= {score_trun} (main): {better_pr_score}"
            )
        _set_output("success", "1" if success else "0")
