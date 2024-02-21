import pytest
import numpy as np
from pyriemann.utils.mean import mean_euclid, mean_logeuclid
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann_qiskit.utils.mean import mean_euclid_cpm, mean_logeuclid_cpm
from pyriemann_qiskit.utils import ClassicalOptimizer, NaiveQAOAOptimizer


def test_performance(get_covmats, get_labels):
    metric = {"mean": "euclid_cpm", "distance": "euclid"}

    clf = make_pipeline(XdawnCovariances(), MDM(metric=metric))
    skf = StratifiedKFold(n_splits=5)
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize(
    "means", [(mean_euclid, mean_euclid_cpm), (mean_logeuclid, mean_logeuclid_cpm)]
)
def test_analytic_vs_cpm_mean(get_covmats, means):
    """Test that analytic and cpm mean returns close results"""
    analytic_mean, cpm_mean = means
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = cpm_mean(covmats)
    C_analytic = analytic_mean(covmats)
    assert np.allclose(C, C_analytic, atol=0.001)


@pytest.mark.parametrize(
    "mean", [mean_euclid_cpm, mean_logeuclid_cpm]
)
def test_mean_cpm_shape(get_covmats, mean):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean(covmats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize("optimizer", [ClassicalOptimizer(), NaiveQAOAOptimizer()])
def test_mean_cpm_all_zeros(optimizer):
    """Test that the mean of covariance matrices containing zeros
    is a matrix filled with zeros"""
    n_trials, n_channels = 5, 2
    covmats = np.zeros((n_trials, n_channels, n_channels))
    C = mean_euclid_cpm(covmats, optimizer=optimizer)
    assert np.allclose(covmats[0], C, atol=0.001)


def test_mean_cpm_all_ones():
    """Test that the mean of covariance matrices containing ones
    is a matrix filled with ones"""
    n_trials, n_channels = 5, 2
    covmats = np.ones((n_trials, n_channels, n_channels))
    C = mean_euclid_cpm(covmats)
    assert np.allclose(covmats[0], C, atol=0.001)


def test_mean_cpm_all_equals():
    """Test that the mean of covariance matrices filled with the same value
    is a matrix identical to the input"""
    n_trials, n_channels, value = 5, 2, 2.5
    covmats = np.full((n_trials, n_channels, n_channels), value)
    C = mean_euclid_cpm(covmats)
    assert np.allclose(covmats[0], C, atol=0.001)


def test_mean_cpm_mixed():
    """Test that the mean of covariances matrices with zero and ones
    is a matrix filled with 0.5"""
    n_trials, n_channels = 5, 2
    covmats_0 = np.zeros((n_trials, n_channels, n_channels))
    covmats_1 = np.ones((n_trials, n_channels, n_channels))
    expected_mean = np.full((n_channels, n_channels), 0.5)
    C = mean_euclid_cpm(np.concatenate((covmats_0, covmats_1), axis=0))
    assert np.allclose(expected_mean, C, atol=0.001)
