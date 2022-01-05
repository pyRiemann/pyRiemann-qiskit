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


def test_quantic_fvt_Classical(get_dataset):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    q = QuanticSVM(quantum=False, verbose=False)
    # We need to have different values for first and second classes
    # in our samples or vector machine will not converge
    n_samples, n_features, n_classes = 100, 9, 2
    class_len = n_samples // n_classes  # balanced set
    samples, labels = get_dataset(n_samples, n_features, n_classes,
                                  random=False)

    q.fit(samples, labels)
    # This will autodefine testing sets
    prediction = q.predict(samples)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == q.classes_[0]
    assert prediction[class_len:].all() == q.classes_[1]


def test_quantic_svm_fvt_simulated_quantum(get_dataset):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    q = QuanticSVM(quantum=True, verbose=False)
    # We need to have different values for target and non-target in our samples
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_samples, n_features, n_classes = 10, 4, 2
    class_len = n_samples // n_classes  # balanced set
    samples, labels = get_dataset(n_samples, n_features, n_classes,
                                  random=False)

    q.fit(samples, labels)
    prediction = q.predict(samples)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == q.classes_[0]
    assert prediction[class_len:].all() == q.classes_[1]


def test_quantic_vqc_fvt_simulated_quantum(get_dataset):
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    q = QuanticVQC(verbose=False)
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_samples, n_features, n_classes = 4, 4, 2
    samples, labels = get_dataset(n_samples, n_features, n_classes)

    q.fit(samples, labels)
    prediction = q.predict(samples)
    # Considering the inputs, this probably make no sense to test accuracy.
    # Instead, we could consider this test as a canary test
    assert len(prediction) == len(labels)
