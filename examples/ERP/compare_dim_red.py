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

from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC, RiemannQuantumClassifier
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

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = dict(vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method="iir")  # replace baselining with high-pass
events = read_events(event_fname)

raw.info["bads"] = ["MEG 2443"]  # set bad channels
picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Read epochs
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)

epochs = epochs[:20]

X = epochs.get_data()
y = epochs.events[:, -1]


###############################################################################
# Decoding in tangent space with a quantum classifier



# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side

# filters = [Closest(ndim=5), Preclassif(ndim=5), NaivePair(), NaiveImpair(), PCA(n_components=5), NoFilter()]


nfilters = [1, 1]

f, axes = plt.subplots(3, len(nfilters), sharey='row')

disp = None

# in common: shots, feature_map
# quanticsvm (gamma, shots, feature_map)
# quanticvqc (shots, feature_map, optimizer, two_local)
# svc? (gamma)

gamma = [0.001, 0.01, 0.1]
shots = [512, 1024, 2048]
feature_entanglement=['full', 'linear', 'circular', 'sca']
reps = [2, 3, 4]

dim_reds = [NaivePair(), NaiveImpair(), NoFilter(), PCA()]

pipe = RiemannQuantumClassifier()
param_grid_qsvc= {
        "nfilter" : [1],
        "gamma" : gamma,
        "dim_red": dim_reds,
        "shots" : shots,
        "feature_entanglement" : feature_entanglement,
        "feature_reps" : reps,
        "spsa_trials" : [None],
        "two_local_reps" : [None]
    }
param_grid_vqc= {
        "nfilter" : [1],
        "gamma" : [None],
        "dim_red": dim_reds,
        "shots" : shots,
        "feature_entanglement" : feature_entanglement,
        "feature_reps" : reps,
        "spsa_trials" : [20, 40, 60],
        "two_local_reps" : reps
    }
param_grid_classical = {
        "nfilter" : [1],
        "gamma" : gamma,
        "dim_red": dim_reds,
        "shots" : [None],
        "feature_entanglement" : feature_entanglement,
        "feature_reps" : reps,
        "spsa_trials" : [None],
        "two_local_reps" : [None]
    }
search_qsvc = GridSearchCV(pipe, param_grid_qsvc, n_jobs=3)
search_qsvc.fit(X, y)

search_vqc = GridSearchCV(pipe, param_grid_vqc, n_jobs=3)
search_vqc.fit(X, y)

search = GridSearchCV(pipe, param_grid_classical, n_jobs=3)
search.fit(X, y)


SVC = 0
QSVC = 1
VQC = 2

# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for classif in [SVC, QSVC, VQC]:
    for i in range(len(nfilters)):
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        
        if classif == QSVC:
            params = search_qsvc.best_params_
            title = "QSVC"
            axe = axes[0][i]
        elif classif == VQC:
            params = search_vqc.best_params_
            title = "VQC"
            axe = axes[1][i]
        else:
            params = search.best_params_
            title = "SVC"
            axe = axes[2][i]

        pipe = RiemannQuantumClassifier(**params)

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

print("Best parameter (CV score=%0.3f):" % search_qsvc.best_score_)
print(search_qsvc.best_params_)
print("Best parameter (CV score=%0.3f):" % search_vqc.best_score_)
print(search_vqc.best_params_)
print("Best parameter Classical (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

if disp:
    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
