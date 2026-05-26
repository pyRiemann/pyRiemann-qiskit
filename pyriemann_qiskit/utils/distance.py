"""Deprecated: use pyriemann_qiskit.optimization.distance instead."""

import warnings

warnings.warn(
    "pyriemann_qiskit.utils.distance is deprecated and will be removed "
    "in a future release. "
    "Use pyriemann_qiskit.optimization.distance instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyriemann_qiskit.optimization.distance import (  # noqa: F401, E402
    qdistance_logeuclid_to_convex_hull,
    weights_logeuclid_to_convex_hull,
)

__all__ = [
    "qdistance_logeuclid_to_convex_hull",
    "weights_logeuclid_to_convex_hull",
]
