"""
====================================================================
TODO
====================================================================

TODO

"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_tangentspace.py
# License: BSD (3-clause)

from typing import Dict
import numpy as np
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.classification import \
    QuantumClassifierWithDefaultRiemannianPipeline
from pyriemann_qiskit.utils.filtering import NaiveDimRed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, \
    cross_val_predict
from sklearn.metrics import balanced_accuracy_score

print(__doc__)

X, y = get_mne_sample(n_trials=-1)

dim_reds = [PCA(n_components=10), NaiveDimRed()]
n_dim_reds = len(dim_reds)


class PCA2(PCA):
    def fit(self, X, y):
        super().n_components = min(X.shape) / 2
        super().fit(X, y)

default = {
    "nfilter": [2],
    "gamma": [None],
    "shots": [1024],
    "feature_entanglement": ['linear'],#['linear', 'sca'],
    "reps": [2], #[2, 3],
    "dim_reds": dim_reds
}

pipe = QuantumClassifierWithDefaultRiemannianPipeline()

def print_score(data, nfilter):
    title = data["title"]
    scores = data["best_score"]
    print("Best %s parameter (nfilter=%i, score=%0.3f):"
            % (title, nfilter + 1, scores[nfilter]), data["best_params"][nfilter])

def get_grid_search(idx, title, a_gamma, a_spsa_trials=[None],
                    a_two_local_reps=[None], a_shots=default["shots"]):
    params = {
        "nfilter": default["nfilter"],
        "gamma": a_gamma,
        "dim_red": default["dim_reds"],
        "shots": a_shots,
        "feature_entanglement": default["feature_entanglement"],
        "feature_reps": default["reps"],
        "spsa_trials": a_spsa_trials,
        "two_local_reps": a_two_local_reps
    }
    grid = GridSearchCV(pipe, params, scoring='balanced_accuracy',n_jobs=-1, cv=StratifiedKFold(n_splits=2))
    search = grid.fit(X, y)
    # seargrid.score(X)
    # search = grid.score(X, y)
    print(search.cv_results_)
    print(search.best_score_)
    
    param_dim_red = search.cv_results_["param_dim_red"]
    scores = search.cv_results_["mean_test_score"]
    test_params = search.cv_results_["params"]

    best_params = []
    best_score = []
    for i in range(n_dim_reds):
        i_best_score = np.where(scores == max(scores[param_dim_red == dim_reds[i]]))
        i_best_score = i_best_score[0][0]
        best_params.append(test_params[i_best_score])
        best_score.append(scores[i_best_score])

    return {
        "idx": idx,
        "title": title,
        "best_params": best_params,
        "best_score": best_score
    }

SVC = get_grid_search(2, "SVC", default["gamma"], [None], [None], [None])
exit(0)
QSVC = get_grid_search(0, "QSVC", default["gamma"])

VQC = get_grid_search(1, "VQC", [None], [40], default["reps"])

_, axes = plt.subplots(n_dim_reds, 1, figsize=(10,7))

for i in range(n_dim_reds):
    scores = [0] * 3
    titles = [""] * 3
    axe = axes[i]
    for classif in [SVC, QSVC, VQC]: 
        params = classif["best_params"][i]
        n = classif["idx"]
        titles[n] = classif["title"]
        scores[n] = classif["best_score"][i]
        print_score(classif, i)
    axe.bar(titles, scores)
    axe.set_title("Balanced accuracies (dimred=%s)" % str(dim_reds[i]))
    

plt.show()