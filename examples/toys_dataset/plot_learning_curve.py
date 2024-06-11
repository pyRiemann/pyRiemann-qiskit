"""
====================================================================
Plot training curve of VQC and MDM with SPSA
====================================================================

In this example, we will show how to plot the training curve
of quantum neural network VQC and MDM, using an SPSA optimizer.

"""

# Author: Gregoire Cattan
# License: BSD (3-clause)

from pyriemann_qiskit.datasets.utils import get_mne_sample
from pyriemann_qiskit.pipelines import QuantumMDMWithRiemannianPipeline
from pyriemann_qiskit.utils.hyper_params_factory import get_spsa
import matplotlib.pyplot as plt
from pyriemann_qiskit.datasets import generate_linearly_separable_dataset
from pyriemann_qiskit.classification import QuanticVQC
from pyriemann.estimation import Shrinkage

print(__doc__)

###############################################################################
# Setup

# Define the plot area
fig, axes = plt.subplots(1, 2)
fig.suptitle("Training curves")

# Generate vectors for VQC
Xv, yv = generate_linearly_separable_dataset(n_samples=20)

# ... and matrices for MDM
Xm, ym = get_mne_sample(n_trials=100)

###############################################################################
# Instantiate the pipelines

# Create the SPSA optimizer
optimizer = get_spsa(max_trials=100)

# Instantiate VQC
vqc = QuanticVQC(optimizer=optimizer)

# ... and the quantum MDM
# This used QAOA under the hood, which is a quantum parametric circuit
# analog to neural network
mdm = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "qlogeuclid", "distance": "logeuclid"},
    quantum=True,
    regularization=Shrinkage(shrinkage=0.9),
    shots=1024,
    seed=696288,
    qaoa_optimizer=optimizer,
)

###############################################################################
# Fit and plot learning curve for VQC

vqc.fit(Xv, yv)
evaluated_values = vqc.evaluated_values_

# Note: The optimizer converge:
# VQC+SPSA is appropriate for the toy dataset.
axe = axes[0]
axe.plot(evaluated_values)
axe.set_ylabel("Evaluated values (VQC)")
axe.set_xlabel("Evaluations")

###############################################################################
# Fit and plot learning curve for MDM with QAOA

mdm.fit(Xm, ym)
evaluated_values = mdm._pipe[2]._optimizer.evaluated_values_

# Note: The optimizer doesn't converge:
# QAOA+SPSA is not appropriate for the MNE dataset.
axe = axes[1]
axe.plot(evaluated_values)
axe.set_ylabel("Evaluated values (MDM)")
axe.set_xlabel("Evaluations")

plt.show()
