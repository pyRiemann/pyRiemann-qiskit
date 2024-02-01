from pyriemann_qiskit.ensemble import (
    JudgeClassifier,
)

def test_canary(estimator):
    assert not JudgeClassifier() is None
