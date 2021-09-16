import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pyriemann.classification import (MDM, FgMDM, KNearestNeighbor,
                                      TSclassifier, QuanticSVM, QuanticVQC)


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_MDM_init():
    """Test init of MDM"""
    MDM(metric='riemann')

    # Should raise if metric not string or dict
    with pytest.raises(TypeError):
        MDM(metric=42)

    # Should raise if metric is not contain bad keys
    with pytest.raises(KeyError):
        MDM(metric={'universe': 42})

    # should works with correct dict
    MDM(metric={'mean': 'riemann', 'distance': 'logeuclid'})


def test_MDM_fit():
    """Test Fit of MDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset, labels)


def test_MDM_predict():
    """Test prediction of MDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    mdm = MDM(metric='riemann')
    mdm.fit(covset, labels)
    mdm.predict(covset)

    # test fit_predict
    mdm = MDM(metric='riemann')
    mdm.fit_predict(covset, labels)

    # test transform
    mdm.transform(covset)

    # predict proba
    mdm.predict_proba(covset)

    # test n_jobs
    mdm = MDM(metric='riemann', n_jobs=2)
    mdm.fit(covset, labels)
    mdm.predict(covset)


def test_KNN():
    """Test KNearestNeighbor"""
    covset = generate_cov(30, 3)
    labels = np.array([0, 1, 2]).repeat(10)

    knn = KNearestNeighbor(1, metric='riemann')
    knn.fit(covset, labels)
    preds = knn.predict(covset)
    assert_array_equal(labels, preds)


def test_TSclassifier():
    """Test TS Classifier"""
    covset = generate_cov(40, 3)
    labels = np.array([0, 1]).repeat(20)

    with pytest.raises(TypeError):
        TSclassifier(clf='666')

    clf = TSclassifier()
    clf.fit(covset, labels)
    assert_array_equal(clf.classes_, np.array([0, 1]))
    clf.predict(covset)
    clf.predict_proba(covset)


def test_FgMDM_init():
    """Test init of FgMDM"""
    FgMDM(metric='riemann')

    # Should raise if metric not string or dict
    with pytest.raises(TypeError):
        FgMDM(metric=42)

    # Should raise if metric is not contain bad keys
    with pytest.raises(KeyError):
        FgMDM(metric={'universe': 42})

    # should works with correct dict
    FgMDM(metric={'mean': 'riemann', 'distance': 'logeuclid'})


def test_FgMDM_predict():
    """Test prediction of FgMDM"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    fgmdm = FgMDM(metric='riemann')
    fgmdm.fit(covset, labels)
    fgmdm.predict(covset)
    fgmdm.transform(covset)

def test_Quantic_init():
    """Test init of quantum classifiers"""
    # if "classical" computation enable, no provider and backend should be defined
    q = QuanticSVM(target=1, quantum=False)
    assert(not q.quantum)
    assert(not hasattr(q, "backend"))
    assert(not hasattr(q, "provider"))
    # if "quantum" computation enabled, but no accountToken are provided, then "quantum" simulation will be enabled 
    # i.e., no remote quantum provider will be defined
    q = QuanticSVM(target=1, quantum=True)
    assert(q.quantum)
    assert(hasattr(q, "backend"))
    assert(not hasattr(q, "provider"))
    # if "quantum" computation enabled, and accountToken is provided, then real quantum backend is used
    # this should raise a error as uncorrect API Token is passed
    try:
        q = QuanticSVM(target=1, quantum=True, qAccountToken="Test")
        assert(False) #Should never reach this line
    except:
        pass

def test_Quantic_splitTargetAndNonTarget():
    """Test splitTargetAndNonTarget method of quantum classifiers"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    q = QuanticSVM(target=1, quantum=False)
    xta, xnt = q.splitTargetAndNonTarget(covset, labels)
    assert(len(xta) == 50)
    # Covariance matrices should be vectorized
    assert(len(xta[0]) == 3 * 3)
    assert(len(xnt) == 50)
    assert(len(xnt[0]) == 3 * 3)

def test_Quantic_SelfCalibration():
    """Test self_calibration method of quantum classifiers"""
    covset = generate_cov(100, 3)
    labels = np.array([0, 1]).repeat(50)
    q = QuanticSVM(target=1, quantum=False)
    q.fit(covset, labels)
    test_size = 0.33 #internal setting to self_calibration method
    len_test = int(test_size * 100)
    # Just using a little trick as fit and score method are called by self_calibration method
    def fit(X_train, y_train):
        assert(len(X_train) == 100 - len_test)
        assert(len(y_train) == 100 - len_test)
        # Covariances matrices of fit and score method should always be non-vectorized
        assert(len(X_train[0]) == 3)
        assert(len(X_train[0][0]) == 3)
    def score(X_test, y_test):
        assert(len(X_test) == len_test)
        assert(len(y_test) == len_test)
        assert(len(X_test[0]) == 3)
        assert(len(X_test[0][0]) == 3)
    q.fit = fit
    q.score = score
    q.self_calibration()

def test_Quantic_FVT_Classical():
    """Perform standard SVC test (canary test to assess pipeline correctness)"""
    # When quantum=False, it should use classical SVC implementation of SKlearn.
    q = QuanticSVM(target=1, quantum=False, verbose=False)
    # We need to have different values for target and non-target in our covset or vector machine will not converge
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
    assert(prediction[0:iNt].all() == 0)
    assert(prediction[iNt:].all() == 1)

def test_QuanticSVM_FVT_SimulatedQuantum():
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time. 
    """
    # We will use a quantum simulator on the local machine
    q = QuanticSVM(target=1, quantum=True, verbose=False)
    # We need to have different values for target and non-target in our covset or vector machine will not converge
    # To achieve testing in a reasonnable amount of time, we will lower the size of the feature and the number of trials
    iNt = 10
    iTa = 5
    nt = np.zeros((iNt, 2, 2))
    ta = np.ones((iTa, 2, 2))
    covset = np.concatenate((nt, ta), axis=0)
    labels = np.concatenate((np.array([0] * iNt), np.array([1] * iTa)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial, therefore we will skip self_calibration
    # as it may happens that self_calibration select only target or non-target trials
    q.test_input = {"Target":[[1,1,1,1]], "NonTarget":[[0,0,0,0]]}
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert(prediction[0:iNt].all() == 0)
    assert(prediction[iNt:].all() == 1)

def test_QuanticVQC_FVT_SimulatedQuantum():
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    q = QuanticVQC(target=1, verbose=False)
    # We need to have different values for target and non-target in our covset or vector machine will not converge
    # To achieve testing in a reasonnable amount of time, we will lower the size of the feature and the number of trials
    iNt = 2
    iTa = 2
    nt = np.zeros((iNt, 2, 2))
    ta = np.ones((iTa, 2, 2))
    covset = np.concatenate((nt, ta), axis=0)
    labels = np.concatenate((np.array([0] * iNt), np.array([1] * iTa)), axis=0)
    q.fit(covset, labels)
    # We are dealing with a small number of trial, therefore we will skip self_calibration
    # as it may happens that self_calibration select only target or non-target trials
    q.test_input = {"Target":[[1,1,1,1]], "NonTarget":[[0,0,0,0]]}
    prediction = q.predict(covset)
    # Considering the inputs, this probably make no sense to test accuracy. Instead, we could consider this test as a canary test
    assert(len(prediction) == len(labels))

