import pytest
import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification \
    import (QuanticSVM,
            QuanticVQC,
            QuantumClassifierWithDefaultRiemannianPipeline)
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.utils.filtering import NaiveDimRed
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from operator import itemgetter


@pytest.mark.parametrize('estimator',
                         [make_pipeline(XdawnCovariances(nfilter=1),
                                        TangentSpace(), NaiveDimRed(),
                                        QuanticSVM(quantum=False)),
                          QuantumClassifierWithDefaultRiemannianPipeline(
                              nfilter=1,
                              shots=None)])
def test_get_set_params(estimator):
    skf = StratifiedKFold(n_splits=2)
    X, y = get_mne_sample()
    scr = cross_val_score(estimator, X, y, cv=skf, scoring='roc_auc',
                          error_score='raise')
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
        n_samples, n_features, \
            quantum_instance, type = itemgetter('n_samples',
                                                'n_features',
                                                'quantum_instance',
                                                'type')(self.get_params())
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
        return {
            "n_samples": 100,
            "n_features": 9,
            "quantum_instance": quantum_instance,
            "type": "rand"
        }

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
        return {
            "n_samples": 100,
            "n_features": 9,
            "quantum_instance": quantum_instance,
            "type": "bin"
        }

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
        return {
            "n_samples": 10,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": "bin"
        }

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
        return {
            "n_samples": 4,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": "rand"
        }

    def check(self):
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(self.prediction) == len(self.labels)


class TestQuantumClassifierWithDefaultRiemannianPipeline(BinaryFVT):
    """Functional testing for riemann quantum classifier."""
    def get_params(self):
        quantum_instance = \
            QuantumClassifierWithDefaultRiemannianPipeline(
                params={'verbose': False}
            )
        return {
            "n_samples": 4,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": None
        }

    def check(self):
        assert len(self.prediction) == len(self.labels)
