from pyriemann_qiskit.utils import union_of_diff
import numpy as np


def test_union_of_diff():
    A = np.array([0, 1, 0])
    B = np.array([1, 1, 0])
    C = np.array([0, 1, 1])
    diff = union_of_diff(A, B, C)
    assert np.array_equal(diff, [True, False, True])
