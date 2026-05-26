"""Deprecated: use pyriemann_qiskit.optimization.mean instead."""

import warnings

warnings.warn(
    "pyriemann_qiskit.utils.mean is deprecated and will be removed "
    "in a future release. "
    "Use pyriemann_qiskit.optimization.mean instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyriemann_qiskit.optimization.mean import (  # noqa: F401, E402
    qmean_euclid,
    qmean_logeuclid,
)

__all__ = [
    "qmean_euclid",
    "qmean_logeuclid",
]
