import pytest
import numpy as np
from pyriemann_qiskit.utils import (ClassicalOptimizer,
                                    NaiveQAOAOptimizer,
                                    logeucl_dist_convex)
from pyriemann.utils.mean import mean_euclid
from pyriemann.utils.distance import distance_methods
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

def test_performance(get_covmats, get_labels):
    metric = {
        'mean': "logeuclid",
        'distance': "convex"
    }

    clf = make_pipeline(XdawnCovariances(), MDM(metric=metric))
    skf = StratifiedKFold(n_splits=5)
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    score = cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')
    assert score.mean() > 0


@pytest.mark.parametrize('optimizer',
                         [ClassicalOptimizer(),
                          NaiveQAOAOptimizer()])
def test_logeucl_dist_convex(optimizer):
    X_0 = np.array([[0.9, 1.1],[0.9, 1.1]])
    X_1 = X_0 + 1
    X = np.stack((X_0, X_1))
    y = (X_0 + X_1) / 3
    weight = logeucl_dist_convex(X, y, optimizer=optimizer)
    assert weight.argmin() == 0 # 0.3 closer to 1
