"""
====================================================================
ERP EEG decoding with Quantum Classifier.
====================================================================

Decoding applied to EEG data in sensor space decomposed using Xdawn.
After spatial filtering, covariances matrices are estimated, then projected in
the tangent space and classified with a quantum SVM classifier.
It is compared to the classical SVM.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_tangentspace.py
# License: BSD (3-clause)

from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.classification import QuantumClassifierWithDefaultRiemannianPipeline
from pyriemann_qiskit.utils.filtering import NaivePair, NaiveImpair, NoFilter
from pyriemann_qiskit.utils.hyper_params_factory import gen_zz_feature_map, gen_two_local, get_spsa
from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import balanced_accuracy_score


print(__doc__)

X, y = get_mne_sample()


###############################################################################
# Decoding in tangent space with a quantum classifier



# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side

# filters = [Closest(ndim=5), Preclassif(ndim=5), NaivePair(), NaiveImpair(), PCA(n_components=5), NoFilter()]


nfilters = [1, 1]

f, axes = plt.subplots(3, len(nfilters), sharey='row')

disp = None

default = {
    "gamma": [0.01],
    "shots": [1024],
    "feature_entanglement": ['linear', 'sca'],
    "reps": [2, 3],
    "dim_reds": [NaivePair(), PCA()]
}

pipe = QuantumClassifierWithDefaultRiemannianPipeline()
idx = -1

def get_grid_search(title, a_gamma, a_spsa_trials=[None], a_two_local_reps=[None], a_shots=default["shots"]):
    params = {
        "nfilter" : [1],
        "gamma" : a_gamma,
        "dim_red": default["dim_reds"],
        "shots" : a_shots,
        "feature_entanglement" : default["feature_entanglement"],
        "feature_reps" : default["reps"],
        "spsa_trials" : a_spsa_trials,
        "two_local_reps" : a_two_local_reps
    }
    grid = GridSearchCV(pipe, params, n_jobs=3)
    search = grid.fit(X, y)
    idx += 1
    return {
        "idx": idx,
        "title": title,
        "best_params": search.best_params_,
        "best_score": search.best_score_
    }


QSVC = get_grid_search("QSVC", default["gamma"])

VQC = get_grid_search("VQC", [None], [40], default["reps"])

SVC = get_grid_search("SVC", default["gamma"], [None], [None], [None])

# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for classif in [SVC, QSVC, VQC]:
    for i in range(len(nfilters)):
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        
        params = classif["best_params"]
        title = classif["title"]
        idx = classif["idx"]
        axe = axes[idx][i]

        pipe = QuantumClassifierWithDefaultRiemannianPipeline(**params)

        y_pred = cross_val_predict(pipe, X, y, cv=cv)

        score = balanced_accuracy_score(y_pred, y)
        # Printing the results
        score_str = "%0.2f" % score

        names = ["vis left", "vis right"]

        title = title + " (" + score_str + ")" + " (nfilter = " + str(nfilters[i]) + ")"
        cm = confusion_matrix(y_pred, y)
        disp = ConfusionMatrixDisplay(cm, display_labels=names)
        disp.plot(ax=axe, xticks_rotation=45)
        disp.ax_.set_title(title)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i > 0:
            disp.ax_.set_ylabel('')
        # if not quantum:
        #     disp.ax_.set_xlabel('')

print("Best parameter (CV score=%0.3f):" % QSVC["best_score"])
print(QSVC["best_params"])
print("Best parameter (CV score=%0.3f):" % VQC["best_score"])
print(VQC["best_params"])
print("Best parameter Classical (CV score=%0.3f):" % SVC["best_score"])
print(SVC["best_params"])

if disp:
    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
