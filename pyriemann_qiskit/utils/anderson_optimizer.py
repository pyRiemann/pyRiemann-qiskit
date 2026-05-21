"""Deprecated: use pyriemann_qiskit.optimization.anderson_optimizer instead."""

import warnings

warnings.warn(
    "pyriemann_qiskit.utils.anderson_optimizer is deprecated and will be removed "
    "in a future release. "
    "Use pyriemann_qiskit.optimization.anderson_optimizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyriemann_qiskit.optimization.anderson_optimizer import (  # noqa: F401, E402
    AndersonAccelerationOptimizer,
)

__all__ = [
    "AndersonAccelerationOptimizer",
]
