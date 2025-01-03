"""
====================================================================
QAOA-CV optimization
====================================================================

QAOA is a parametric quantum circuit, which is usually used to
solve QUBO problems, i.e, problems with binary variables [1]_.

In this example we will show how to use pyRiemann-qiskit implementation
of QAOA-CV, which accepts continuous variables.

This example demonstrates how to use the QAOA-CV optimizer
on a simple objective function.

"""

import math

import matplotlib.pyplot as plt
from docplex.mp.model import Model
from qiskit_algorithms.optimizers import COBYLA, SPSA

from pyriemann_qiskit.utils.docplex import QAOACVOptimizer
from pyriemann_qiskit.utils.hyper_params_factory import (
    create_mixer_qiskit_default,
    create_mixer_rotational_X_gates,
    create_mixer_rotational_XY_gates,
    create_mixer_rotational_XZ_gates,
)

###############################################################################
# Run QAOA-CV
#
# Let's define a handy function to run and plot the result of the QAOA-CV


def run_qaoa_cv(n_reps, optimizer, create_mixer):
    # Define docplex model
    mdl = Model("docplex model")
    # Domain of definition for the variables
    x = mdl.continuous_var(-1, 0, "x")
    y = mdl.continuous_var(0, 2, "y")
    z = mdl.continuous_var(1.1, 2.2, "z")
    # objective function to minimize
    mdl.minimize((x - 0.83 + y + 2 * z) ** 2)

    # Instanciate the QAOA-CV
    # Note: if quantum_instance is None, it will be created inside the optimizer.
    qaoa_cv = QAOACVOptimizer(
        create_mixer, n_reps, quantum_instance=None, optimizer=optimizer
    )

    # reshape is when working with covariance matrices
    # So the vector of solution is reshaped into a matrix
    # (this is not the case here)
    solution = qaoa_cv.solve(mdl, reshape=False)

    # print the time, the solution (that it the value for our three variable)
    # and the minimum of the objective function
    print(f"time = {qaoa_cv.run_time_}")
    print(f"solution = {solution}")
    print(f"min = {qaoa_cv.minimum_}")

    # Display the loss function
    plt.plot(qaoa_cv.x_, qaoa_cv.y_)

    # And the state vector of the optimized quantum circuit
    print(qaoa_cv.state_vector_)

    plt.show()

    return (qaoa_cv.run_time_, qaoa_cv.minimum_)


###############################################################################
# Hyper-parameters
#
# QAOA is a quantum circuit that is a repetition of
# a cost an mixer operator.
#
# We will now try different combination of optimizer and mixers.

maxiter = 500
optimizers = [
    COBYLA(maxiter=maxiter),
    SPSA(maxiter=maxiter),
]

n_angles = 4
n_repetitions = 5

ret = {}

for angle in range(n_angles):
    angle = math.pi * angle / n_angles
    mixers = [
        create_mixer_qiskit_default(angle),
        create_mixer_rotational_X_gates(angle),
        create_mixer_rotational_XY_gates(angle),
        create_mixer_rotational_XZ_gates(angle),
    ]
    for opt in optimizers:
        for rep in range(1, n_repetitions):
            for create_mixer in mixers:
                print(
                    f"Running QAOA with angle {angle}, \
                    optimizer {type(opt).__name__}, \
                    {rep} repetitions and\
                    {create_mixer.__name__} method"
                )
                key = f"{angle}_{type(opt).__name__}_{rep}_{create_mixer.__name__}"
                ret[key] = run_qaoa_cv(rep, opt, create_mixer)


###############################################################################
# References
# ----------
# .. [1] https://dice.cyfronet.pl/papers/JPlewa_JSienko_msc_v2.pdf
