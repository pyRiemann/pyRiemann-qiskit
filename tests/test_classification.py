import numpy as np
from pyriemann_qiskit.classification import (QuanticSVM, QuanticVQC)


def test_Quantic_init():
    """Test init of quantum classifiers"""
    # if "classical" computation enable,
    # no provider and backend should be defined
    q = QuanticSVM(target=1, quantum=False)
    assert not q.quantum
    assert not hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, but no accountToken are provided,
    # then "quantum" simulation will be enabled
    # i.e., no remote quantum provider will be defined
    q = QuanticSVM(target=1, quantum=True)
    assert q.quantum
    assert hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, and accountToken is provided,
    # then real quantum backend is used
    # this should raise a error as uncorrect API Token is passed
    try:
        q = QuanticSVM(target=1, quantum=True, qAccountToken="Test")
        assert False  # Should never reach this line
    except Exception:
        pass


def test_Quantic_splitTargetAndNonTarget(get_covmats):
    """Test splitTargetAndNonTarget method of quantum classifiers"""
    n_matrices, n_channels = 100, 3
    nt, ta = 0, 1
    covset = get_covmats(n_matrices, n_channels)
    labels = np.array([nt, ta]).repeat(n_matrices // 2)
    q = QuanticSVM(target=1, quantum=False)
    xta, xnt = q.splitTargetAndNonTarget(covset, labels)
    assert len(xta) == n_matrices // 2
    # Covariance matrices should be vectorized
    assert len(xta[0]) == n_channels * n_channels
    assert len(xnt) == n_matrices // 2
    assert len(xnt[0]) == n_channels * n_channels


def test_Quantic_SelfCalibration(get_covmats):
    """Test self_calibration method of quantum classifiers"""
    n_matrices, n_channels = 100, 3
    nt, ta = 0, 1
    covset = get_covmats(n_matrices, n_channels)
    labels = np.array([nt, ta]).repeat(n_matrices // 2)
    q = QuanticSVM(target=ta, quantum=False)
    q.fit(covset, labels)
    test_size = 0.33  # internal setting to self_calibration method
    len_test = int(test_size * n_matrices)
    # Just using a little trick as fit and score method are
    # called by self_calibration method

    def fit(X_train, y_train):
        assert len(X_train) == n_matrices - len_test
        assert len(y_train) == n_matrices - len_test
        # Covariances matrices of fit and score method
        # should always be non-vectorized
        assert len(X_train[0]) == n_channels
        assert len(X_train[0][0]) == n_channels

    def score(X_test, y_test):
        assert len(X_test) == len_test
        assert len(y_test) == len_test
        assert len(X_test[0]) == n_channels
        assert len(X_test[0][0]) == n_channels
    q.fit = fit
    q.score = score
    q.self_calibration()


def test_Quantic_FVT_Classical():
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    nt, ta = 0, 1
    q = QuanticSVM(target=ta, quantum=False, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    n_matrices, n_channels = 100, 3
    n_nt = 3 * n_matrices // 4
    n_ta = n_matrices // 4
    nt_covset = np.zeros((n_nt, 3, 3))
    ta_covset = np.ones((n_ta, 3, 3))
    covset = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = np.concatenate((np.array([0] * n_nt), np.array([1] * n_ta)), axis=0)
    q.fit(covset, labels)
    # This will autodefine testing sets
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[0:n_nt].all() == nt
    assert prediction[n_nt:].all() == ta


def test_QuanticSVM_FVT_SimulatedQuantum():
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    nt, ta = 0, 1
    q = QuanticSVM(target=ta, quantum=True, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_nt = 10
    n_ta = 5
    nt_covset = np.zeros((n_nt, 2, 2))
    ta_covset = np.ones((n_ta, 2, 2))
    covset = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = np.concatenate((np.array([nt] * n_nt), np.array([ta] * n_ta)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens
    # that self_calibration select only target or non-target trials
    q.test_input = {"Target": [[ta] * 4], "NonTarget": [[nt] * 4]}
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[0:n_nt].all() == nt
    assert prediction[n_nt:].all() == ta


def test_QuanticVQC_FVT_SimulatedQuantum():
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    nt, ta = 0, 1
    q = QuanticVQC(target=1, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_nt = 2
    n_ta = 2
    nt_covset = np.zeros((n_nt, 2, 2))
    ta_covset = np.ones((n_ta, 2, 2))
    covset = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = np.concatenate((np.array([nt] * n_nt), np.array([ta] * n_ta)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens that
    # self_calibration select only target or non-target trials
    q.test_input = {"Target": [[ta] * 4], "NonTarget": [[nt] * 4]}
    prediction = q.predict(covset)
    # Considering the inputs, this probably make no sense to test accuracy.
    # Instead, we could consider this test as a canary test
    assert len(prediction) == len(labels)
