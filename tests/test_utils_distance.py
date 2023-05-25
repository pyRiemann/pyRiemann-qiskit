import pytest
import numpy as np
from pyriemann_qiskit.utils import (ClassicalOptimizer,
                                    NaiveQAOAOptimizer,
                                    logeucl_dist_convex)


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
