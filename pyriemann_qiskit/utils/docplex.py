"""Deprecated: use pyriemann_qiskit.optimization.docplex instead."""

import warnings

warnings.warn(
    "pyriemann_qiskit.utils.docplex is deprecated and will be removed "
    "in a future release. "
    "Use pyriemann_qiskit.optimization.docplex instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyriemann_qiskit.optimization.docplex import (  # noqa: F401, E402
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
    QAOACVAngleOptimizer,
    QAOACVOptimizer,
    build_qaoa_ansatz,
    pyQiskitOptimizer,
    square_bin_mat_var,
    square_cont_mat_var,
    square_int_mat_var,
)

__all__ = [
    "square_cont_mat_var",
    "square_int_mat_var",
    "square_bin_mat_var",
    "pyQiskitOptimizer",
    "ClassicalOptimizer",
    "NaiveQAOAOptimizer",
    "build_qaoa_ansatz",
    "QAOACVAngleOptimizer",
    "QAOACVOptimizer",
]
