import pytest
import numpy as np
from pyriemann.utils.mean import mean_euclid, mean_logeuclid
from pyriemann.estimation import XdawnCovariances, Shrinkage
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann_qiskit.utils.mean import qmean_euclid, qmean_logeuclid
from pyriemann_qiskit.utils import ClassicalOptimizer, NaiveQAOAOptimizer
from pyriemann_qiskit.classification import QuanticMDM
from pyriemann_qiskit.datasets import get_mne_sample
from qiskit_optimization.algorithms import ADMMOptimizer


@pytest.mark.parametrize(
    "kernel",
    [
        ({"mean": "qeuclid", "distance": "euclid"}, Shrinkage(shrinkage=0.9)),
        ({"mean": "qlogeuclid", "distance": "logeuclid"}, Shrinkage(shrinkage=0.9)),
    ],
)
def test_performance(kernel):
    metric, regularization = kernel
    clf = make_pipeline(
        XdawnCovariances(),
        QuanticMDM(metric=metric, regularization=regularization, quantum=False),
    )
    skf = StratifiedKFold(n_splits=3)

    covset, labels = get_mne_sample()

    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize(
    "means", [(mean_euclid, qmean_euclid), (mean_logeuclid, qmean_logeuclid)]
)
def test_analytic_vs_cpm_mean(get_covmats, means):
    """Test that analytic and cpm mean returns close results"""
    analytic_mean, cpm_mean = means
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = cpm_mean(covmats)
    C_analytic = analytic_mean(covmats)
    assert np.allclose(C, C_analytic, atol=0.00001)


@pytest.mark.parametrize("mean", [qmean_euclid, qmean_logeuclid])
def test_mean_cpm_shape(get_covmats, mean):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = mean(covmats)
    assert C.shape == (n_channels, n_channels)


@pytest.mark.parametrize(
    "optimizer",
    [ClassicalOptimizer(optimizer=ADMMOptimizer()), NaiveQAOAOptimizer()],
)
@pytest.mark.parametrize("mean", [qmean_euclid])
def test_mean_cpm_all_zeros(optimizer, mean):
    """Test that the mean of covariance matrices containing zeros
    is a matrix filled with zeros"""
    n_trials, n_channels = 5, 2
    covmats = np.zeros((n_trials, n_channels, n_channels))
    C = mean(covmats, optimizer=optimizer)
    assert np.allclose(covmats[0], C, atol=0.001)


@pytest.mark.parametrize(
    "optimizer",
    [ClassicalOptimizer(optimizer=ADMMOptimizer()), NaiveQAOAOptimizer()],
)
@pytest.mark.parametrize("mean", [qmean_euclid])
def test_mean_cpm_all_ones(optimizer, mean):
    """Test that the mean of covariance matrices containing ones
    is a matrix filled with ones"""
    n_trials, n_channels = 5, 2
    covmats = np.ones((n_trials, n_channels, n_channels))
    C = mean(covmats, optimizer=optimizer)
    assert np.allclose(covmats[0], C, atol=0.001)


@pytest.mark.parametrize(
    "optimizer",
    [ClassicalOptimizer(optimizer=ADMMOptimizer()), NaiveQAOAOptimizer()],
)
@pytest.mark.parametrize("mean", [qmean_euclid])
def test_mean_cpm_all_equals(optimizer, mean):
    """Test that the mean of covariance matrices filled with the same value
    is a matrix identical to the input"""
    n_trials, n_channels, value = 5, 2, 2.5
    covmats = np.full((n_trials, n_channels, n_channels), value)
    C = mean(covmats, optimizer=optimizer)
    assert np.allclose(covmats[0], C, atol=0.001)


@pytest.mark.parametrize(
    "optimizer",
    [ClassicalOptimizer(optimizer=ADMMOptimizer()), NaiveQAOAOptimizer()],
)
@pytest.mark.parametrize("mean", [qmean_euclid])
def test_mean_cpm_mixed(optimizer, mean):
    """Test that the mean of covariances matrices with zero and ones
    is a matrix filled with 0.5"""
    n_trials, n_channels = 5, 2
    covmats_0 = np.zeros((n_trials, n_channels, n_channels))
    covmats_1 = np.ones((n_trials, n_channels, n_channels))
    expected_mean = np.full((n_channels, n_channels), 0.5)
    C = mean(np.concatenate((covmats_0, covmats_1), axis=0), optimizer=optimizer)
    assert np.allclose(expected_mean, C, atol=0.001)
