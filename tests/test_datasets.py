import numpy as np
import pytest
from pyriemann.estimation import Covariances

from pyriemann_qiskit.utils.dataset import (
    generate_subject_signal,
    get_feature_dimension,
)


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


@pytest.mark.parametrize(
    "n_trials_per_class, n_channels, n_times, n_classes",
    [(5, 3, 32, 2), (1, 2, 8, 2), (10, 4, 64, 4)],
)
def test_generate_subject_signal_shapes(
    n_trials_per_class, n_channels, n_times, n_classes
):
    X, y = generate_subject_signal(
        n_trials_per_class, n_channels, n_times, n_classes, subj_seed=42
    )
    assert X.shape == (n_trials_per_class * n_classes, n_channels, n_times)
    assert y.shape == (n_trials_per_class * n_classes,)
    assert np.isfinite(X).all()


def test_generate_subject_signal_labels_are_balanced():
    n_trials_per_class, n_classes = 7, 3
    _, y = generate_subject_signal(n_trials_per_class, 4, 32, n_classes, subj_seed=42)
    classes, counts = np.unique(y, return_counts=True)
    np.testing.assert_array_equal(classes, np.arange(n_classes))
    np.testing.assert_array_equal(counts, np.full(n_classes, n_trials_per_class))


def test_generate_subject_signal_is_deterministic():
    """Same `subj_seed` must reproduce a subject exactly, and different
    seeds must give different channel mixing -- that mixing is what
    simulates the subject-specific domain shift the transfer-learning
    examples rely on.
    """
    args = (5, 3, 32, 2)
    X1, y1 = generate_subject_signal(*args, subj_seed=7)
    X2, y2 = generate_subject_signal(*args, subj_seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

    X3, y3 = generate_subject_signal(*args, subj_seed=8)
    assert not np.allclose(X1, X3)
    np.testing.assert_array_equal(y1, y3)


def _class_cov_gap(scale_factor, subj_seed):
    """Relative Frobenius gap between the two class-mean covariance matrices.

    Normalized by the covariance magnitude so the measure is scale-free and
    the threshold below does not depend on the generator's overall variance.
    """
    X, y = generate_subject_signal(
        40, 3, 512, 2, subj_seed=subj_seed, scale_factor=scale_factor
    )
    covs = Covariances(estimator="lwf").fit_transform(X)
    mean_0, mean_1 = covs[y == 0].mean(axis=0), covs[y == 1].mean(axis=0)
    return np.linalg.norm(mean_0 - mean_1) / np.linalg.norm(mean_0 + mean_1)


@pytest.mark.parametrize("subj_seed", [0, 1, 2])
def test_generate_subject_signal_scale_factor_drives_class_separability(subj_seed):
    """The generator's contract: the class signal must survive covariance
    estimation, and `scale_factor` is what controls its strength.

    `scale_factor=1.0` scales no channel differently per class, so both
    classes share one covariance model and the gap must vanish; raising it
    must open a gap. Thresholds are far from the observed values (measured
    <= 0.016 at 1.0 and >= 0.47 at 2.0 across seeds), so this asserts the
    mechanism, not a tuned number.
    """
    gap_none = _class_cov_gap(1.0, subj_seed)
    gap_default = _class_cov_gap(2.0, subj_seed)

    assert gap_none < 0.1
    assert gap_default > 0.25
    assert gap_default > gap_none


def test_generate_subject_signal_rejects_more_classes_than_channels():
    """Class `cls` is encoded by scaling channel `cls`, so the documented
    n_classes <= n_channels constraint is a hard requirement.
    """
    with pytest.raises(ValueError, match="n_classes must be <= n_channels"):
        generate_subject_signal(4, 2, 32, 3, subj_seed=42)
