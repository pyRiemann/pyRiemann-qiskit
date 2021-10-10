import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def test_GetSetParams(get_covmats, get_labels):
    clf = make_pipeline(XdawnCovariances(), TangentSpace(),
                        QuanticSVM(quantum=False))
    skf = StratifiedKFold(n_splits=5)
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')


def test_qsvm_init():
    """Test init of quantum classifiers"""
    # if "classical" computation enable,
    # no provider and backend should be defined
    q = QuanticSVM(quantum=False)
    q._init_quantum()
    assert not q.quantum
    assert not hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, but no accountToken are provided,
    # then "quantum" simulation will be enabled
    # i.e., no remote quantum provider will be defined
    q = QuanticSVM(quantum=True)
    q._init_quantum()
    assert q.quantum
    assert hasattr(q, "_backend")
    assert not hasattr(q, "_provider")
    # if "quantum" computation enabled, and accountToken is provided,
    # then real quantum backend is used
    # this should raise a error as uncorrect API Token is passed
    try:
        q = QuanticSVM(labelsquantum=True, qAccountToken="Test")
        assert False  # Should never reach this line
    except Exception:
        pass


def test_qsvm_splitclasses(get_2d_covmats, get_labels):
    """Test _split_classes method of quantum classifiers"""
    q = QuanticSVM(quantum=False)

    n_matrices, n_channels, n_classes = 100, 3, 2
    covset = get_2d_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    # As fit method is not called here, classes_ is not set.
    # so we need to provide the classes ourselves.
    q.classes_ = range(0, n_classes)

    x_class1, x_class0 = q._split_classes(covset, labels)
    # Covariance matrices should be vectorized
    class_len = n_matrices // n_classes  # balanced set
    assert np.shape(x_class1) == (class_len, n_channels * n_channels)
    assert np.shape(x_class0) == (class_len, n_channels * n_channels)


def test_qsvm_selfcalibration(get_2d_covmats, get_labels):
    """Test _self_calibration method of quantum classifiers"""

    test_size = 0.33
    q = QuanticSVM(quantum=False, test_per=test_size)

    n_matrices, n_channels, n_classes = 100, 3, 2
    covset = get_2d_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    len_test = int(test_size * n_matrices)

    q.fit(covset, labels)
    # Just using a little trick as fit and score method are
    # called by self_calibration method

    def test_fit(X_train, y_train):
        assert len(y_train) == n_matrices - len_test
        # Covariances matrices of fit and score method
        # should always be non-vectorized
        assert X_train.shape == (n_matrices - len_test,
                                 n_channels * n_channels)

    def test_score(X_test, y_test):
        assert len(y_test) == len_test
        assert X_test.shape == (len_test, n_channels * n_channels)

    q.fit = test_fit
    q.score = test_score
    q._self_calibration()


def test_Quantic_FVT_Classical(get_labels):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    q = QuanticSVM(quantum=False, verbose=False)
    # We need to have different values for first and second classes
    # in our covset or vector machine will not converge
    n_matrices, n_channels, n_classes = 100, 3, 2
    class_len = n_matrices // n_classes  # balanced set
    covset_0 = np.zeros((class_len, n_channels * n_channels))
    covset_1 = np.ones((class_len, n_channels * n_channels))
    covset = np.concatenate((covset_0, covset_1), axis=0)
    labels = get_labels(n_matrices, n_classes)

    q.fit(covset, labels)
    # This will autodefine testing sets
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == q.classes_[0]
    assert prediction[class_len:].all() == q.classes_[1]


def test_QuanticSVM_FVT_SimulatedQuantum(get_labels):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    q = QuanticSVM(quantum=True, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 10, 2, 2
    class_len = n_matrices // n_classes  # balanced set
    covset_0 = np.zeros((class_len, n_channels * n_channels))
    covset_1 = np.ones((class_len, n_channels * n_channels))
    covset = np.concatenate((covset_0, covset_1), axis=0)
    labels = get_labels(n_matrices, n_classes)

    q.fit(covset, labels)
    prediction = q.predict(covset)
    # In this case, using SVM, predicting accuracy should be 100%
    assert prediction[:class_len].all() == q.classes_[0]
    assert prediction[class_len:].all() == q.classes_[1]


def test_QuanticVQC_FVT_SimulatedQuantum(get_2d_covmats, get_labels):
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    q = QuanticVQC(verbose=False)
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 4, 2, 2
    covset = get_2d_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    q.fit(covset, labels)
    prediction = q.predict(covset)
    # Considering the inputs, this probably make no sense to test accuracy.
    # Instead, we could consider this test as a canary test
    assert len(prediction) == len(labels)
