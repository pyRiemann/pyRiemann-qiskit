from pyriemann_qiskit.utils.docplex import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
    QAOACVAngleOptimizer,
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
    qaoa_initial_points,
    qaoacv_implementation,
):
    """Factory function to create optimizer based on parameters.

    Creates and returns the appropriate optimizer instance (quantum or classical)
    based on the provided parameters. Selects between NaiveQAOAOptimizer,
    QAOACVAngleOptimizer, QAOACVOptimizer, or ClassicalOptimizer depending on
    the configuration.

    Parameters
    ----------
    logger : object
        Logger object with _log method for logging optimizer selection.
    quantum : bool
        If True, creates a quantum optimizer. If False, creates a classical optimizer.
    quantum_instance : QuantumInstance or None
        Quantum instance for running quantum circuits. Required when quantum=True.
    upper_bound : int
        Upper bound for integer variables in NaiveQAOAOptimizer.
    qaoa_optimizer : Optimizer
        Classical optimizer for QAOA parameter optimization.
    classical_optimizer : Optimizer
        Classical optimizer instance used when quantum=False.
    create_mixer : callable or None
        Function to create custom mixer for QAOA-CV. If None and quantum is True, uses NaiveQAOAOptimizer.
    n_reps : int
        Number of repetitions for QAOA ansatz layers.
    qaoa_initial_points : array-like
        Initial parameter values for QAOA optimization.
    qaoacv_implementation : str
        QAOA-CV implementation variant. "ulvi" selects QAOACVAngleOptimizer,
        "luna" or other values select QAOACVOptimizer. If not quantum or create_mixer is undefined, then does nothing.

    Returns
    -------
    optimizer : ClassicalOptimizer, NaiveQAOAOptimizer, QAOACVAngleOptimizer, or QAOACVOptimizer
        Configured optimizer instance based on the provided parameters.

    Notes
    -----
    The function selects the optimizer according to the following logic:
    - If quantum=False: returns ClassicalOptimizer
    - If quantum=True and create_mixer is None: returns NaiveQAOAOptimizer
    - If quantum=True and create_mixer is provided:
        - If "ulvi" in qaoacv_implementation: returns QAOACVAngleOptimizer
        - Otherwise: returns QAOACVOptimizer

    Notes
    -----
    .. versionadded:: 0.4.1
    .. versionchanged:: 0.5.0
            add qaoacv_implementation parameter

    """
    if quantum:
        if create_mixer:
            if "ulvi" in qaoacv_implementation:
                logger._log("Using QAOACVAngleOptimizer")
                return QAOACVAngleOptimizer(
                    create_mixer=create_mixer,
                    n_reps=n_reps,
                    quantum_instance=quantum_instance,
                    optimizer=qaoa_optimizer,
                )
            else:
                if not "luna" in qaoacv_implementation:
                    logger._log("No valid QAOA-CV implementation found.")
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
                initial_points=qaoa_initial_points,
            )
    else:
        logger._log(f"Using ClassicalOptimizer ({type(classical_optimizer).__name__})")
        return ClassicalOptimizer(classical_optimizer)
