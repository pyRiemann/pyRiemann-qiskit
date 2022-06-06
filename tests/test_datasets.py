import pytest
from pyriemann_qiskit.datasets import get_feature_dimension


def test_get_feature_dimension_fvt(get_dataset):
    n_samples, n_features, n_classes = 100, 10, 2
    samples, labels = get_dataset(n_samples, n_features, n_classes)
    dataset = {}
    dataset[0] = samples[labels == 0]
    dataset[1] = samples[labels == 1]
    assert get_feature_dimension(dataset) == n_features


def test_get_feature_dimension_with_wrong_type_raises_error():
    with pytest.raises(TypeError):
        get_feature_dimension(None)


def test_get_feature_dimension_with_empty_dataset_returns_unvalid_dimension():
    assert get_feature_dimension({}) == -1
