"""Deprecated: use pyriemann_qiskit.utils.dataset instead."""

import warnings

warnings.warn(
    "pyriemann_qiskit.datasets is deprecated and will be removed in a future release. "
    "Use pyriemann_qiskit.utils.dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyriemann_qiskit.utils.dataset import (  # noqa: F401, E402
    get_mne_sample,
    generate_qiskit_dataset,
    generate_linearly_separable_dataset,
    get_feature_dimension,
    MockDataset,
)

__all__ = [
    "get_mne_sample",
    "generate_qiskit_dataset",
    "generate_linearly_separable_dataset",
    "get_feature_dimension",
    "MockDataset",
]
