import pytest
import numpy as np
from pyriemann_qiskit.utils import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
)
from pyriemann_qiskit.utils.distance import weights_logeuclid_to_convex_hull_cpm
from pyriemann_qiskit.classification import QuanticMDM
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


@pytest.mark.parametrize(
    "kernel",
    [
        ({"mean": "euclid", "distance": "euclid_cpm"}, None),
        ({"mean": "logeuclid", "distance": "logeuclid_hull_cpm"}, None),
    ],
)
def test_performance(kernel):
    metric, regularization = kernel

    clf = make_pipeline(XdawnCovariances(), QuanticMDM(metric=metric, quantum=False, regularization=regularization))
    skf = StratifiedKFold(n_splits=3)
    covset, labels = get_mne_sample()
    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize("optimizer", [ClassicalOptimizer(), NaiveQAOAOptimizer()])
def test_distance_logeuclid_to_convex_hull_cpm(optimizer):
    X_0 = np.array([[0.9, 1.1], [0.9, 1.1]])
    X_1 = X_0 + 1
    X = np.stack((X_0, X_1))
    y = (X_0 + X_1) / 3
    weights = weights_logeuclid_to_convex_hull_cpm(X, y, optimizer=optimizer)
    distances = 1 - weights
    assert distances.argmin() == 0
