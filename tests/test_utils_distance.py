import pytest
import numpy as np
from pyriemann_qiskit.utils import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
)
from pyriemann_qiskit.utils.distance import (
    qdistance_logeuclid_to_convex_hull,
    weights_logeuclid_to_convex_hull,
)
from pyriemann_qiskit.classification import QuanticMDM
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.mean import mean_logeuclid
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


@pytest.mark.parametrize(
    "metric",
    [
        {"mean": "euclid", "distance": "qeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid_hull"},
    ],
)
def test_performance(metric):
    clf = make_pipeline(XdawnCovariances(), QuanticMDM(metric=metric, quantum=False))
    skf = StratifiedKFold(n_splits=3)
    covset, labels = get_mne_sample()
    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize("optimizer", [ClassicalOptimizer(), NaiveQAOAOptimizer()])
def test_qdistance_logeuclid_to_convex_hull(optimizer, get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)

    dist = qdistance_logeuclid_to_convex_hull(covmats, covmats[0], optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)

    covmean = mean_logeuclid(covmats)
    dist = qdistance_logeuclid_to_convex_hull(covmats, covmean, optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("optimizer", [ClassicalOptimizer(), NaiveQAOAOptimizer()])
def test_weight_logeuclid_to_convex_hull(optimizer):
    X_0 = np.array([[0.9, 1.1], [0.9, 1.1]])
    X_1 = X_0 + 1
    X_train = np.stack((X_0, X_1))
    X_test = (X_0 + X_1) / 3
    weights = weights_logeuclid_to_convex_hull(X_train, X_test, optimizer=optimizer)
    distances = 1 - weights
    assert distances.argmin() == 0
