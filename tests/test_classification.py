from conftest import get_distances, get_means, get_metrics
import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.classification import MDM, FgMDM, KNearestNeighbor, TSclassifier, QuanticSVM, QuanticVQC
from pyriemann.estimation import Covariances
import pytest
from pytest import approx
from sklearn.dummy import DummyClassifier


rclf = [MDM, FgMDM, KNearestNeighbor, TSclassifier]


@pytest.mark.parametrize("classif", rclf)
class ClassifierTestCase:
    def test_two_classes(self, classif, get_covmats, get_labels):
        n_classes, n_trials, n_channels = 2, 6, 3
        covmats = get_covmats(n_trials, n_channels)
        labels = get_labels(n_trials, n_classes)
        self.clf_predict(classif, covmats, labels)
        self.clf_fit_independence(classif, covmats, labels)
        if classif is MDM:
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if classif in (MDM, FgMDM, KNearestNeighbor):
            self.clf_jobs(classif, covmats, labels)
        if classif in (MDM, FgMDM, TSclassifier):
            self.clf_predict_proba(classif, covmats, labels)
            self.clf_populate_classes(classif, covmats, labels)
        if classif is KNearestNeighbor:
            self.clf_predict_proba_trials(classif, covmats, labels)
        if classif is (FgMDM, TSclassifier):
            self.clf_tsupdate(classif, covmats, labels)

    def test_multi_classes(self, classif, get_covmats, get_labels):
        n_classes, n_trials, n_channels = 3, 9, 3
        covmats = get_covmats(n_trials, n_channels)
        labels = get_labels(n_trials, n_classes)
        self.clf_fit_independence(classif, covmats, labels)
        self.clf_predict(classif, covmats, labels)
        if classif is MDM:
            self.clf_fitpredict(classif, covmats, labels)
        if classif in (MDM, FgMDM):
            self.clf_transform(classif, covmats, labels)
        if classif in (MDM, FgMDM, KNearestNeighbor):
            self.clf_jobs(classif, covmats, labels)
        if classif in (MDM, FgMDM, TSclassifier):
            self.clf_predict_proba(classif, covmats, labels)
            self.clf_populate_classes(classif, covmats, labels)
        if classif is KNearestNeighbor:
            self.clf_predict_proba_trials(classif, covmats, labels)
        if classif is (FgMDM, TSclassifier):
            self.clf_tsupdate(classif, covmats, labels)


class TestClassifier(ClassifierTestCase):
    def clf_predict(self, classif, covmats, labels):
        n_trials = len(labels)
        clf = classif()
        clf.fit(covmats, labels)
        predicted = clf.predict(covmats)
        assert predicted.shape == (n_trials,)

    def clf_predict_proba(self, classif, covmats, labels):
        n_trials = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        probabilities = clf.predict_proba(covmats)
        assert probabilities.shape == (n_trials, n_classes)
        assert probabilities.sum(axis=1) == approx(np.ones(n_trials))

    def clf_predict_proba_trials(self, classif, covmats, labels):
        n_trials = len(labels)
        # n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        probabilities = clf.predict_proba(covmats)
        assert probabilities.shape == (n_trials, n_trials)
        assert probabilities.sum(axis=1) == approx(np.ones(n_trials))

    def clf_fitpredict(self, classif, covmats, labels):
        clf = classif()
        clf.fit_predict(covmats, labels)
        assert_array_equal(clf.classes_, np.unique(labels))

    def clf_transform(self, classif, covmats, labels):
        n_trials = len(labels)
        n_classes = len(np.unique(labels))
        clf = classif()
        clf.fit(covmats, labels)
        transformed = clf.transform(covmats)
        assert transformed.shape == (n_trials, n_classes)

    def clf_fit_independence(self, classif, covmats, labels):
        clf = classif()
        clf.fit(covmats, labels).predict(covmats)
        # retraining with different size should erase previous fit
        new_covmats = covmats[:, :-1, :-1]
        clf.fit(new_covmats, labels).predict(new_covmats)

    def clf_jobs(self, classif, covmats, labels):
        clf = classif(n_jobs=2)
        clf.fit(covmats, labels)
        clf.predict(covmats)

    def clf_populate_classes(self, classif, covmats, labels):
        clf = classif()
        clf.fit(covmats, labels)
        assert_array_equal(clf.classes_, np.unique(labels))

    def clf_classif_tsupdate(self, classif, covmats, labels):
        clf = classif(tsupdate=True)
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", [MDM, FgMDM, TSclassifier])
@pytest.mark.parametrize("mean", ["faulty", 42])
@pytest.mark.parametrize("dist", ["not_real", 27])
def test_metric_dict_error(classif, mean, dist, get_covmats, get_labels):
    with pytest.raises((TypeError, KeyError)):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = classif(metric={"mean": mean, "distance": dist})
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", [MDM, FgMDM])
@pytest.mark.parametrize("mean", get_means())
@pytest.mark.parametrize("dist", get_distances())
def test_metric_dist(classif, mean, dist, get_covmats, get_labels):
    n_trials, n_channels, n_classes = 4, 3, 2
    labels = get_labels(n_trials, n_classes)
    covmats = get_covmats(n_trials, n_channels)
    clf = classif(metric={"mean": mean, "distance": dist})
    clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
def test_metric_wrong_keys(classif, metric, get_covmats, get_labels):
    with pytest.raises((TypeError, KeyError)):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = classif(metric=metric)
        clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("metric", get_metrics())
def test_metric_str(classif, metric, get_covmats, get_labels):
    n_trials, n_channels, n_classes = 6, 3, 2
    labels = get_labels(n_trials, n_classes)
    covmats = get_covmats(n_trials, n_channels)
    clf = classif(metric=metric)
    clf.fit(covmats, labels).predict(covmats)


@pytest.mark.parametrize("dist", ["not_real", 42])
def test_knn_dict_dist(dist, get_covmats, get_labels):
    with pytest.raises(KeyError):
        n_trials, n_channels, n_classes = 6, 3, 2
        labels = get_labels(n_trials, n_classes)
        covmats = get_covmats(n_trials, n_channels)
        clf = KNearestNeighbor(metric={"distance": dist})
        clf.fit(covmats, labels).predict(covmats)


def test_1NN(get_covmats, get_labels):
    """Test KNearestNeighbor with K=1"""
    n_trials, n_channels, n_classes = 9, 3, 3
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)

    knn = KNearestNeighbor(1, metric="riemann")
    knn.fit(covmats, labels)
    preds = knn.predict(covmats)
    assert_array_equal(labels, preds)


def test_TSclassifier_classifier(get_covmats, get_labels):
    """Test TS Classifier"""
    n_trials, n_channels, n_classes = 6, 3, 2
    covmats = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    clf = TSclassifier(clf=DummyClassifier())
    clf.fit(covmats, labels).predict(covmats)


def test_TSclassifier_classifier_error():
    """Test TS if not Classifier"""
    with pytest.raises(TypeError):
        TSclassifier(clf=Covariances())


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


def test_Quantic_splitTargetAndNonTarget(get_covmats):
    """Test splitTargetAndNonTarget method of quantum classifiers"""
    covset = get_covmats(100, 3)
    labels = np.array([0, 1]).repeat(50)
    q = QuanticSVM(target=1, quantum=False)
    xta, xnt = q.splitTargetAndNonTarget(covset, labels)
    assert(len(xta) == 50)
    # Covariance matrices should be vectorized
    assert(len(xta[0]) == 3 * 3)
    assert(len(xnt) == 50)
    assert(len(xnt[0]) == 3 * 3)


def test_Quantic_SelfCalibration(get_covmats):
    """Test self_calibration method of quantum classifiers"""
    covset = get_covmats(100, 3)
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
