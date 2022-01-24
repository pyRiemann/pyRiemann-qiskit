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

import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticSVM
from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
import itertools


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

X = epochs.get_data()
y = epochs.events[:, -1]


###############################################################################
# Decoding in tangent space with a quantum classifier

# Time complexity of quantum algorithm depends on the number of trials and
# the number of elements inside the covariance matrices
# Thus we reduce elements number by using restrictive spatial filtering
sf = XdawnCovariances(nfilter=2)

class NoFilter(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class NaivePair(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, ::2]

class NaiveImpair(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, 1::2]

class Closest(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        print(indices, self.ndim)
        for v in X:
            for subset in itertools.combinations(indices, self.ndim):
                not_indices = [i for i in indices if i not in subset]
                sub_v = v.copy()
                sub_v[not_indices] = 0
                dist = np.linalg.norm(v - sub_v)
                key = ''.join(str(val) for val in subset)
                if key in ret:
                    ret[key] = ret[key] + dist
                else:
                    ret[key] = dist
        self.closest_subset_to_vector = min(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret

# ...and dividing the number of remaining elements by two
class Preclassif(TransformerMixin):
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.closest_subset_to_vector = None

    def fit(self, X, y=None):
        n_ts = X.shape[1]
        if self.ndim is None:
            self.ndim = n_ts // 2
        indices = range(n_ts)
        ret = {}
        for subset in itertools.combinations(indices, self.ndim):
            sub_vectors = None
            for v in X:
                if sub_vectors is None:
                    sub_vectors = np.atleast_2d([v[list(subset)]])
                else:
                    sub_vectors = np.append(sub_vectors, [v[list(subset)]], axis=0)
            svc = SVC()

            svc.fit(sub_vectors, y)
            key = ''.join(str(val) for val in subset)
            ret[key] = svc.score(sub_vectors, y)
        self.closest_subset_to_vector = max(ret, key=ret.get)
        print("Fit finished")
        return self

    def transform(self, X, y=None):
        subset = [int(i) for i in self.closest_subset_to_vector]
        ret = X[:, subset]
        return ret


# Projecting correlation matrices into the tangent space
# as quantum algorithms take vectors as inputs
# (If not, then matrices will be inlined inside the quantum classifier)
tg = TangentSpace()

# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side


# filters = [Closest(ndim=5), Preclassif(ndim=5), NaivePair(), NaiveImpair(), PCA(n_components=5), NoFilter()]
filters = [NaivePair(), PCA(n_components=5)]
n_filters = len(filters)

f, axes = plt.subplots(2, n_filters, sharey='row')

disp = None

# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for quantum in [False, True]:
    for i in range(n_filters):

        filter = filters[i]
        
        filter_name = type(filter).__name__

        print("running filter " + filter_name)
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        
        clf = make_pipeline(sf, tg, filter, QuanticSVM(quantum=quantum))
        # score = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy').mean()
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        score = balanced_accuracy_score(y_pred, y)
        # Printing the results
        score_str = "%0.2f" % score

        # names = ['0', '1']
        names = ["vis left", "vis right"]
        # names = ["vis left", "vis right"]

        title = filter_name + "(" + score_str + ")"
        axe = axes[1 if quantum else 0][i]
        cm = confusion_matrix(y_pred, y)
        disp = ConfusionMatrixDisplay(cm, display_labels=names)
        disp.plot(ax=axe, xticks_rotation=45)
        disp.ax_.set_title(title)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i > 0:
            disp.ax_.set_ylabel('')
        if not quantum:
            disp.ax_.set_xlabel('')

if disp:
    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
