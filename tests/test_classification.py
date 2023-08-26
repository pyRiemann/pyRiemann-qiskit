import pytest
from conftest import BinaryTest, BinaryFVT, MultiLabelsFVT, MultiLabelsTest
import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import (
    QuanticSVM,
    QuanticVQC,
    QuanticMDM,
)
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.utils.filtering import NaiveDimRed
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


@pytest.mark.parametrize(
    "estimator",
    [
        make_pipeline(
            XdawnCovariances(nfilter=1),
            TangentSpace(),
            NaiveDimRed(),
            QuanticSVM(quantum=False),
        ),
        make_pipeline(XdawnCovariances(nfilter=1), QuanticMDM(quantum=False)),
    ],
)
def test_get_set_params(estimator):
    skf = StratifiedKFold(n_splits=2)
    X, y = get_mne_sample()
    scr = cross_val_score(
        estimator, X, y, cv=skf, scoring="roc_auc", error_score="raise"
    )
    assert scr.mean() > 0


def test_vqc_classical_should_return_value_error():
    with pytest.raises(ValueError):
        QuanticVQC(quantum=False)


def test_qsvm_init_quantum_wrong_token():
    with pytest.raises(Exception):
        q = QuanticSVM(quantum=True, q_account_token="INVALID")
        q._init_quantum()


@pytest.mark.parametrize("quantum", [False, True])
def test_qsvm_init(quantum):
    """Test init of quantum classifiers"""

    q = QuanticSVM(quantum=quantum)
    q._init_quantum()
    assert q.quantum == quantum
    assert hasattr(q, "_backend") == quantum

    # A provider is only assigned when running on a real quantum backend
    assert not hasattr(q, "_provider")


class TestQSVMSplitClasses(BinaryTest):
    """Test _split_classes method of quantum classifiers"""

    def get_params(self):
        quantum_instance = QuanticSVM(quantum=False)
        return {
            "n_samples": 100,
            "n_features": 9,
            "quantum_instance": quantum_instance,
            "type": "rand",
        }

    def additional_steps(self):
        # As fit method is not called here, classes_ is not set.
        # so we need to provide the classes ourselves.
        self.quantum_instance.classes_ = range(0, self.n_classes)
        self.x_classes = self.quantum_instance._split_classes(self.samples, self.labels)

    def check(self):
        for i in range(self.n_classes):
            assert np.shape(self.x_classes[i]) == (self.class_len, self.n_features)


class TestQSVMSplitClasses_MultiLabels(MultiLabelsTest):
    """Test _split_classes method of quantum classifiers (with 3 classes)"""

    def get_params(self):
        return TestQSVMSplitClasses.get_params(self)

    def additional_steps(self):
        return TestQSVMSplitClasses.additional_steps(self)

    def check(self):
        return TestQSVMSplitClasses.check(self)


class TestClassicalSVM(BinaryFVT):
    """Perform functional validation testing of Quantic SVM"""

    def get_params(self):
        quantum_instance = QuanticSVM(quantum=False, verbose=False)
        return {
            "n_samples": 100,
            "n_features": 9,
            "quantum_instance": quantum_instance,
            "type": "bin",
        }

    def check(self):
        assert (
            self.prediction[: self.class_len].all() == self.quantum_instance.classes_[0]
        )
        assert (
            self.prediction[self.class_len :].all() == self.quantum_instance.classes_[1]
        )


class TestQuanticSVM(TestClassicalSVM):
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
            "type": "bin",
        }


class TestQuanticPegasosSVM(TestClassicalSVM):
    """Same as TestQuanticSVM, expect it uses
    PegasosQSVC instead of QSVC implementation.
    """

    def get_params(self):
        quantum_instance = QuanticSVM(quantum=True, verbose=False, pegasos=True)
        return {
            "n_samples": 10,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": "bin",
        }


class TestQuanticVQC(BinaryFVT):
    """Perform VQC on a simulated quantum computer"""

    def get_params(self):
        quantum_instance = QuanticVQC(verbose=False)
        # To achieve testing in a reasonnable amount of time,
        # we will lower the size of the feature and the number of trials
        return {
            "n_samples": 6,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": "rand",
        }

    def check(self):
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(self.prediction) == len(self.labels)
        # Check the number of classes is consistent
        assert len(np.unique(self.prediction)) == len(np.unique(self.labels))


class TestQuanticVQC_MultiLabels(MultiLabelsFVT):
    """Perform VQC on a simulated quantum computer
    (multi labels classification)"""

    def get_params(self):
        # multi-inheritance pattern
        return TestQuanticVQC.get_params(self)

    def check(self):
        TestQuanticVQC.check(self)


class TestClassicalMDM(BinaryFVT):
    """Test the classical version of MDM inside QuanticMDM wrapper."""

    def get_params(self):
        quantum_instance = QuanticMDM(
            quantum=False,
            verbose=False,
            metric={"mean": "logdet", "distance": "logdet"},
        )
        return {
            "n_samples": 100,
            "n_features": 9,
            "quantum_instance": quantum_instance,
            "type": "bin_cov",
        }

    def check(self):
        assert (
            self.prediction[: self.class_len].all() == self.quantum_instance.classes_[0]
        )
        assert (
            self.prediction[self.class_len :].all() == self.quantum_instance.classes_[1]
        )
