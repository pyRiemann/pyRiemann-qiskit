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
    covset = get_covmats(100, 3)
    labels = np.array([0, 1]).repeat(50)
    q = QuanticSVM(target=1, quantum=False)
    xta, xnt = q.splitTargetAndNonTarget(covset, labels)
    assert len(xta) == 50
    # Covariance matrices should be vectorized
    assert len(xta[0]) == 3 * 3
    assert len(xnt) == 50
    assert len(xnt[0]) == 3 * 3 


def test_Quantic_SelfCalibration(get_covmats):
    """Test self_calibration method of quantum classifiers"""
    covset = get_covmats(100, 3)
    labels = np.array([0, 1]).repeat(50)
    q = QuanticSVM(target=1, quantum=False)
    q.fit(covset, labels)
    test_size = 0.33  # internal setting to self_calibration method
    len_test = int(test_size * 100)
    # Just using a little trick as fit and score method are
    # called by self_calibration method

    def fit(X_train, y_train):
        assert len(X_train) == 100 - len_test
        assert len(y_train) == 100 - len_test
        # Covariances matrices of fit and score method
        # should always be non-vectorized
        assert len(X_train[0]) == 3
        assert len(X_train[0][0]) == 3

    def score(X_test, y_test):
        assert len(X_test) == len_test
        assert len(y_test) == len_test
        assert len(X_test[0]) == 3
        assert len(X_test[0][0]) == 3
    q.fit = fit
    q.score = score
    q.self_calibration()


def test_Quantic_FVT_Classical():
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    q = QuanticSVM(target=1, quantum=False, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    iNt = 75
    iTa = 25
    nt = np.zeros((iNt, 3, 3))
    ta = np.ones((iTa, 3, 3))
    covset = np.concatenate((nt, ta), axis=0)
    labels = np.concatenate((np.array([0]*75), np.array([1]*25)), axis=0)
    q.fit(covset, labels)
    # This will autodefine testing sets
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[0:iNt].all() == 0
    assert prediction[iNt:].all() == 1


def test_QuanticSVM_FVT_SimulatedQuantum():
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    q = QuanticSVM(target=1, quantum=True, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    iNt = 10
    iTa = 5
    nt = np.zeros((iNt, 2, 2))
    ta = np.ones((iTa, 2, 2))
    covset = np.concatenate((nt, ta), axis=0)
    labels = np.concatenate((np.array([0] * iNt), np.array([1] * iTa)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens
    # that self_calibration select only target or non-target trials
    q.test_input = {"Target": [[1, 1, 1, 1]], "NonTarget": [[0, 0, 0, 0]]}
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[0:iNt].all() == 0
    assert prediction[iNt:].all() == 1


def test_QuanticVQC_FVT_SimulatedQuantum():
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    q = QuanticVQC(target=1, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    iNt = 2
    iTa = 2
    nt = np.zeros((iNt, 2, 2))
    ta = np.ones((iTa, 2, 2))
    covset = np.concatenate((nt, ta), axis=0)
    labels = np.concatenate((np.array([0] * iNt), np.array([1] * iTa)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens that
    # self_calibration select only target or non-target trials
    q.test_input = {"Target": [[1, 1, 1, 1]], "NonTarget": [[0, 0, 0, 0]]}
    prediction = q.predict(covset)
    # Considering the inputs, this probably make no sense to test accuracy.
    # Instead, we could consider this test as a canary test
    assert len(prediction) == len(labels)
