"""
====================================================================
TODO
====================================================================

TODO

"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_tangentspace.py
# License: BSD (3-clause)

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

X, y = get_mne_sample(n_trials=10)

nfilters = [1, 2]

f, axes = plt.subplots(len(nfilters), 3, sharey='row')

disp = None

default = {
    "gamma": [0.01],
    "shots": [1024],
    "feature_entanglement": ['linear', 'sca'],
    "reps": [2, 3],
    "dim_reds": [NaiveDimRed(), PCA()]
}

pipe = QuantumClassifierWithDefaultRiemannianPipeline()


def get_grid_search(idx, title, a_gamma, a_spsa_trials=[None],
                    a_two_local_reps=[None], a_shots=default["shots"]):
    params = {
        "nfilter": nfilters,
        "gamma": a_gamma,
        "dim_red": default["dim_reds"],
        "shots": a_shots,
        "feature_entanglement": default["feature_entanglement"],
        "feature_reps": default["reps"],
        "spsa_trials": a_spsa_trials,
        "two_local_reps": a_two_local_reps
    }
    grid = GridSearchCV(pipe, params, scoring='balanced_accuracy',n_jobs=3, cv=StratifiedKFold())
    search = grid.fit(X, y)
    filters = search.cv_results_["param_nfilter"]
    scores = search.cv_results_["mean_test_score"]
    test_params = search.cv_results_["params"]

    i_best_score_1 = np.where(scores == max(scores[filters == 2]))
    i_best_score_2 = np.where(scores == max(scores[filters == 2]))
    i_best_score_1 = i_best_score_1[0][0]
    i_best_score_2 = i_best_score_2[0][0]
    # print(search.cv_results_["rank_test_score"])
    # print(filters)
    # print(scores)
    # print(scores[filters == 1])
    # print(max(scores[filters == 1]))
    print("i_best_score_1", i_best_score_1, i_best_score_1)
    return {
        "idx": idx,
        "title": title,
        "best_params": [test_params[i_best_score_1], test_params[i_best_score_2]],
        "best_score": [scores[i_best_score_1], scores[i_best_score_2]]

    }

SVC = get_grid_search(2, "SVC", default["gamma"], [None], [None], [None])

# print(SVC)
# exit(0)
QSVC = get_grid_search(0, "QSVC", default["gamma"])

VQC = get_grid_search(1, "VQC", [None], [40], default["reps"])




# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for classif in [SVC, QSVC, VQC]:
    for i in range(len(nfilters)):
        params = classif["best_params"][i]
        title = classif["title"]
        n = classif["idx"]
        axe = axes[i][n]


        score = classif["best_score"][i]
        print(classif)
        # Printing the results
        score_str = "%0.2f" % score

        names = ["vis left", "vis right"]

        title = title + " (" + score_str + ")" + \
            " (nfilter = " + str(nfilters[i]) + ")"
        cm = confusion_matrix(y, y)
        disp = ConfusionMatrixDisplay(cm, display_labels=names)
        disp.plot(ax=axe, xticks_rotation=45)
        disp.ax_.set_title(title)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if n > 0:
            disp.ax_.set_ylabel('')
        if i < 2:
            disp.ax_.set_xlabel('')
        # if not quantum:
        #     disp.ax_.set_xlabel('')

print("Best QSVC parameter (CV score=%0.3f):" % QSVC["best_score"][0])
print(QSVC["best_params"])
print("Best VQC parameter (CV score=%0.3f):" % VQC["best_score"][0])
print(VQC["best_params"])
print("Best SVC parameter Classical (CV score=%0.3f):" % SVC["best_score"][0])
print(SVC["best_params"])

if disp:
    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
