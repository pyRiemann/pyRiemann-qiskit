import numpy as np
from pyriemann_qiskit.classification import (QuanticSVM, QuanticVQC)


def test_Quantic_init():
    """Test init of quantum classifiers"""
    ta = 1
    # if "classical" computation enable,
    # no provider and backend should be defined
    q = QuanticSVM(target=ta, quantum=False)
    assert not q.quantum
    assert not hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, but no accountToken are provided,
    # then "quantum" simulation will be enabled
    # i.e., no remote quantum provider will be defined
    q = QuanticSVM(target=ta, quantum=True)
    assert q.quantum
    assert hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, and accountToken is provided,
    # then real quantum backend is used
    # this should raise a error as uncorrect API Token is passed
    try:
        q = QuanticSVM(target=ta, quantum=True, qAccountToken="Test")
        assert False  # Should never reach this line
    except Exception:
        pass


def test_Quantic_splitTargetAndNonTarget(get_covmats, get_labels):
    """Test splitTargetAndNonTarget method of quantum classifiers"""
    n_matrices, n_channels, n_classes = 100, 3, 2
    nt, ta = 0, 1
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    q = QuanticSVM(target=ta, quantum=False)
    xta, xnt = q.splitTargetAndNonTarget(covset, labels)
    # Covariance matrices should be vectorized
    class_len = n_matrices // n_classes # balanced set
    assert np.shape(xta) == (class_len, n_channels * n_channels)
    assert np.shape(xnt) == (class_len, n_channels * n_channels)


def test_Quantic_SelfCalibration(get_covmats, get_labels):
    """Test self_calibration method of quantum classifiers"""
    n_matrices, n_channels, n_classes = 100, 3, 2
    nt, ta = 0, 1
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    q = QuanticSVM(target=ta, quantum=False)
    q.fit(covset, labels)
    test_size = 0.33  # internal setting to self_calibration method
    len_test = int(test_size * n_matrices)
    # Just using a little trick as fit and score method are
    # called by self_calibration method

    def fit(X_train, y_train):
        assert len(y_train) == n_matrices - len_test
        # Covariances matrices of fit and score method
        # should always be non-vectorized
        assert X_train.shape == (n_matrices - len_test, n_channels, n_channels)

    def score(X_test, y_test):
        assert len(y_test) == len_test
        assert X_test.shape == (len_test, n_channels, n_channels)

    q.fit = fit
    q.score = score
    q.self_calibration()


def test_Quantic_FVT_Classical(get_labels):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    nt, ta = 0, 1
    q = QuanticSVM(target=ta, quantum=False, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    n_matrices, n_channels, n_classes = 100, 3, 2
    class_len = n_matrices // n_classes # balanced set
    nt_covset = np.zeros((class_len, n_channels, n_channels))
    ta_covset = np.ones((class_len, n_channels, n_channels))
    covset = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = get_labels(n_matrices, n_classes)
    q.fit(covset, labels)
    # This will autodefine testing sets
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == nt
    assert prediction[class_len:].all() == ta


def test_QuanticSVM_FVT_SimulatedQuantum(get_labels):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    nt, ta = 0, 1
    n_training = 4
    q = QuanticSVM(target=ta, quantum=True, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 10, 2, 2
    class_len = n_matrices // n_classes # balanced set
    nt_covset = np.zeros((class_len, n_channels, n_channels))
    ta_covset = np.ones((class_len, n_channels, n_channels))
    covset = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = get_labels(n_matrices, n_classes)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens
    # that self_calibration select only target or non-target trials
    q.test_input = {"Target": [[ta] * n_training], "NonTarget": [[nt] * n_training]}
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == nt
    assert prediction[class_len:].all() == ta


def test_QuanticVQC_FVT_SimulatedQuantum(get_covmats, get_labels):
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    nt, ta = 0, 1
    n_training = 4
    q = QuanticVQC(target=1, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 4, 2, 2
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens that
    # self_calibration select only target or non-target trials
    q.test_input = {"Target": [[ta] * n_training], "NonTarget": [[nt] * n_training]}
    prediction = q.predict(covset)
    # Considering the inputs, this probably make no sense to test accuracy.
    # Instead, we could consider this test as a canary test
    assert len(prediction) == len(labels)
