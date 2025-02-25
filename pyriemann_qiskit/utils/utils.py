from pyriemann_qiskit.utils.docplex import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
    QAOACVOptimizer,
)

from .distance import distance_functions
from .mean import mean_functions


def is_qfunction(string):
    """Indicates if the function is a mean or a distance introduced in this library.

    Return True is "string" represents a
    mean or a distance introduced in this library.

    Parameters
    ----------
    string: str
        A string representation of the mean/distance.

    Returns
    -------
    is_qfunction : boolean
        True if "string" represents a mean or a distance introduced in this library.

    Notes
    -----
    .. versionadded:: 0.2.0

    """
    return string[0] == "q" and (
        (string in mean_functions) or (string in distance_functions)
    )


def get_docplex_optimizer_from_params_bag(
    logger,
    quantum,
    quantum_instance,
    upper_bound,
    qaoa_optimizer,
    classical_optimizer,
    create_mixer,
    n_reps,
):
    if quantum:
        if create_mixer:
            logger._log("Using QAOACVOptimizer")
            return QAOACVOptimizer(
                create_mixer=create_mixer,
                n_reps=n_reps,
                quantum_instance=quantum_instance,
                optimizer=qaoa_optimizer,
            )
        else:
            logger._log("Using NaiveQAOAOptimizer")
            return NaiveQAOAOptimizer(
                quantum_instance=quantum_instance,
                upper_bound=upper_bound,
                optimizer=qaoa_optimizer,
                initial_points=initi,
            )
    else:
        logger._log(f"Using ClassicalOptimizer ({type(classical_optimizer).__name__})")
        return ClassicalOptimizer(classical_optimizer)
