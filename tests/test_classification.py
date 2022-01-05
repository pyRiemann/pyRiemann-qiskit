import pytest
import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from qiskit.providers.ibmq.api.exceptions import RequestsApiError


def test_params(get_dataset):
    clf = make_pipeline(XdawnCovariances(), TangentSpace(),
                        QuanticSVM(quantum=False))
    skf = StratifiedKFold(n_splits=5)
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset, labels = get_dataset(n_matrices, n_channels, n_classes)

    cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')


def test_vqc_classical_should_return_value_error():
    with pytest.raises(ValueError):
        QuanticVQC(quantum=False)


def test_qsvm_init_quantum_wrong_token():
    with pytest.raises(RequestsApiError):
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


def test_qsvm_splitclasses(get_dataset):
    """Test _split_classes method of quantum classifiers"""
    q = QuanticSVM(quantum=False)

    n_samples, n_features, n_classes = 100, 9, 2
    samples, labels = get_dataset(n_samples, n_features, n_classes)

    # As fit method is not called here, classes_ is not set.
    # so we need to provide the classes ourselves.
    q.classes_ = range(0, n_classes)

    x_class1, x_class0 = q._split_classes(samples, labels)
    class_len = n_samples // n_classes  # balanced set
    assert np.shape(x_class1) == (class_len, n_features)
    assert np.shape(x_class0) == (class_len, n_features)


class BinFVT:
    def prepare(self, n_samples, n_features, quantum_instance, random):
        self.n_classes = 2
        self.n_samples = n_samples
        self.n_features = n_features
        self.quantum_instance = quantum_instance
        self.random = random
        self.class_len = n_samples // self.n_classes

    def test(self, get_dataset):
        # there is no __init__ method with pytest
        n_samples, n_features, quantum_instance, random = self.get_params()
        self.prepare(n_samples, n_features, quantum_instance, random)
        self.samples, self.labels = get_dataset(self.n_samples,
                                                self.n_features,
                                                self.n_classes,
                                                self.random)
        self.quantum_instance.fit(self.samples, self.labels)
        prediction = self.quantum_instance.predict(self.samples)
        self.check(prediction)

    def get_params(self):
        raise NotImplementedError

    def check(self, prediction):
        raise NotImplementedError


class TestClassicalSVM(BinFVT):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    def get_params(self):
        quantum_instance = QuanticSVM(quantum=False, verbose=False)
        return 100, 9, quantum_instance, False

    def check(self, prediction):
        assert prediction[:self.class_len].all() == \
               self.quantum_instance.classes_[0]
        assert prediction[self.class_len:].all() == \
               self.quantum_instance.classes_[1]


class TestQuanticSVM(BinFVT):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    def get_params(self):
        quantum_instance = QuanticSVM(quantum=True, verbose=False)
        return 10, 4, quantum_instance, False

    def check(self, prediction):
        assert prediction[:self.class_len].all() == \
               self.quantum_instance.classes_[0]
        assert prediction[self.class_len:].all() == \
               self.quantum_instance.classes_[1]


class TestQuanticVQC(BinFVT):
    """Perform VQC on a simulated quantum computer"""
    def get_params(self):
        quantum_instance = QuanticVQC(verbose=False)
        # To achieve testing in a reasonnable amount of time,
        # we will lower the size of the feature and the number of trials
        return 4, 4, quantum_instance, True

    def check(self, prediction):
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(prediction) == len(self.labels)
