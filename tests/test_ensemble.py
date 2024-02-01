import pytest
from conftest import BinaryFVT
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from pyriemann_qiskit.datasets import get_mne_sample
from sklearn.model_selection import StratifiedKFold, cross_val_score

def test_canary(estimator):
    assert True
