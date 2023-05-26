import pytest
import numpy as np
from pyriemann_qiskit.utils import (ClassicalOptimizer,
                                    NaiveQAOAOptimizer,
                                    logeucl_dist_convex)
from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def test_performance():
    metric = {
        'mean': "euclid",
        'distance': "convex"
    }

    clf = make_pipeline(XdawnCovariances(), MDM(metric=metric))
    skf = StratifiedKFold(n_splits=5)
    covset, labels = get_mne_sample()
    score = cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')
    assert score.mean() > 0


@pytest.mark.parametrize('optimizer',
                         [ClassicalOptimizer(),
                          NaiveQAOAOptimizer()])
def test_logeucl_dist_convex(optimizer):
    X_0 = np.array([[0.9, 1.1], [0.9, 1.1]])
    X_1 = X_0 + 1
    X = np.stack((X_0, X_1))
    y = (X_0 + X_1) / 3
    distances = logeucl_dist_convex(X, y, optimizer=optimizer)
    assert distances.argmin() == 0
