import pytest
from conftest import BinaryFVT
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
)
from pyriemann_qiskit.datasets import get_mne_sample
from sklearn.model_selection import StratifiedKFold, cross_val_score


@pytest.mark.parametrize(
    "estimator",
    [
        QuantumClassifierWithDefaultRiemannianPipeline(nfilter=1, shots=None),
    ],
)
def test_get_set_params(estimator):
    skf = StratifiedKFold(n_splits=2)
    X, y = get_mne_sample()
    scr = cross_val_score(
        estimator, X, y, cv=skf, scoring="roc_auc", error_score="raise"
    )
    assert scr.mean() > 0


class TestQuantumClassifierWithDefaultRiemannianPipeline(BinaryFVT):
    """Functional testing for riemann quantum classifier."""

    def get_params(self):
        quantum_instance = QuantumClassifierWithDefaultRiemannianPipeline(
            params={"verbose": False, "use_fidelity_state_vector_kernel": False}
        )
        return {
            "n_samples": 4,
            "n_features": 4,
            "quantum_instance": quantum_instance,
            "type": None,
        }

    def check(self):
        assert len(self.prediction) == len(self.labels)
