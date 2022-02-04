from distutils.log import error
from pyriemann_qiskit.utils.filtering import NaiveDimRed
import pytest
import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.spatialfilters import Xdawn
from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC, RiemannQuantumClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_moons, make_circles, make_classification
from qiskit.ml.datasets import ad_hoc_data


from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample


data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = dict(vis_l=3, vis_r=4)  # select only two classes

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

X = epochs.get_data()[:10]
y = epochs.events[:, -1][:10]
    

def test_get_set_params(get_dataset):
    clf = make_pipeline(XdawnCovariances(nfilter=1), TangentSpace(), NaiveDimRed(),
                        QuanticSVM(quantum=False))
    skf = StratifiedKFold(n_splits=2)
    # n_matrices, n_channels, n_classes = 100, 3, 2
    # covset, labels = get_dataset(n_matrices, n_channels, n_classes, "covset")

    scr = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc', error_score='raise')

    assert scr.mean() > 0

def test_params_2(get_dataset):
    clf = RiemannQuantumClassifier(nfilter=1, shots=None)
    skf = StratifiedKFold(n_splits=2)
    # n_matrices, n_channels, n_classes = 100, 2, 2
    # covset, labels = get_dataset(n_matrices, n_channels, n_classes, "")
    scr = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc', error_score='raise')
    
    assert scr.mean() > 0

def test_vqc_classical_should_return_value_error():
    with pytest.raises(ValueError):
        QuanticVQC(quantum=False)


def test_qsvm_init_quantum_wrong_token():
    with pytest.raises(Exception):
        q = QuanticSVM(quantum=True, q_account_token="INVALID")
        q._init_quantum()


@pytest.mark.parametrize('quantum', [False, True])
def test_qsvm_init(quantum):
    """Test init of quantum classifiers"""

    q = QuanticSVM(quantum=quantum)
    q._init_quantum()
    assert q.quantum == quantum
    assert hasattr(q, "_backend") == quantum

    # A provider is only assigned when running on a real quantum backend
    assert not hasattr(q, "_provider")


class BinaryTest:
    def prepare(self, n_samples, n_features, quantum_instance, type):
        self.n_classes = 2
        self.n_samples = n_samples
        self.n_features = n_features
        self.quantum_instance = quantum_instance
        self.type = type
        self.class_len = n_samples // self.n_classes

    def test(self, get_dataset):
        # there is no __init__ method with pytest
        n_samples, n_features, quantum_instance, type = self.get_params()
        self.prepare(n_samples, n_features, quantum_instance, type)
        self.samples, self.labels = get_dataset(self.n_samples,
                                                self.n_features,
                                                self.n_classes,
                                                self.type)
        self.additional_steps()
        self.check()

    def get_params(self):
        raise NotImplementedError

    def additional_steps(self):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError


class TestQSVMSplitClasses(BinaryTest):
    """Test _split_classes method of quantum classifiers"""
    def get_params(self):
        quantum_instance = QuanticSVM(quantum=False)
        return 100, 9, quantum_instance, "rand_feats"

    def additional_steps(self):
        # As fit method is not called here, classes_ is not set.
        # so we need to provide the classes ourselves.
        self.quantum_instance.classes_ = range(0, self.n_classes)
        self.x_class1, \
            self.x_class0 = self.quantum_instance._split_classes(self.samples,
                                                                 self.labels)

    def check(self):
        assert np.shape(self.x_class1) == (self.class_len, self.n_features)
        assert np.shape(self.x_class0) == (self.class_len, self.n_features)


class BinaryFVT(BinaryTest):
    def additional_steps(self):
        self.quantum_instance.fit(self.samples, self.labels)
        self.prediction = self.quantum_instance.predict(self.samples)


class TestClassicalSVM(BinaryFVT):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    def get_params(self):
        quantum_instance = QuanticSVM(quantum=False, verbose=False)
        return 100, 9, quantum_instance, "bin_feats"

    def check(self):
        assert self.prediction[:self.class_len].all() == \
               self.quantum_instance.classes_[0]
        assert self.prediction[self.class_len:].all() == \
               self.quantum_instance.classes_[1]


class TestQuanticSVM(BinaryFVT):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    def get_params(self):
        quantum_instance = QuanticSVM(quantum=True, verbose=False)
        return 10, 4, quantum_instance, "bin_feats"

    def check(self):
        assert self.prediction[:self.class_len].all() == \
               self.quantum_instance.classes_[0]
        assert self.prediction[self.class_len:].all() == \
               self.quantum_instance.classes_[1]


class TestQuanticVQC(BinaryFVT):
    """Perform VQC on a simulated quantum computer"""
    def get_params(self):
        quantum_instance = QuanticVQC(verbose=False)
        # To achieve testing in a reasonnable amount of time,
        # we will lower the size of the feature and the number of trials
        return 4, 4, quantum_instance, "rand_feats"

    def check(self):
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(self.prediction) == len(self.labels)

class TestRiemannQuantumClassifier(BinaryFVT):
    """Functional testing for riemann quantum classifier."""
    def get_params(self):
        quantum_instance = RiemannQuantumClassifier(verbose=False)
        return 4, 4, quantum_instance, ""

    def check(self):
        assert len(self.prediction) == len(self.labels)