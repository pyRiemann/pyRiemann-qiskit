"""
====================================================================
Art visualization of Variational Quantum Classifier.
====================================================================

Display the variability of the weights inside the variational quantum
classifier.

"""

# Author: Gregoire Cattan
# License: BSD (3-clause)

from pyriemann_qiskit.utils.hyper_params_factory import gen_two_local
import matplotlib.pyplot as plt
from pyriemann_qiskit.datasets import get_linearly_separable_dataset
from pyriemann_qiskit.classification import QuanticVQC
from pyriemann_qiskit.visualization import weights_spiral


print(__doc__)

###############################################################################
# In this example we will display weights variability of the parameter inside 
# the variational quantum circuit which is used by VQC.
# 
# The idea is simple :
# - We initialize a VQC with different number of parameters and number of samples
# - We train the VQC a couple of time and we store the fitted weights.
# - We compute variability of the weight and display it in a fashion way.

# Let's start by defining some plot area.
fig, axes = plt.subplots(2, 2)
fig.suptitle('VQC weights variability')

# We will compute weight variability for different number of samples
for i, n_samples in enumerate([2, 5]):
    # ... and for differente number of parameters.
    # (n_reps controls the number of parameters inside the circuit)
    for j, n_reps in enumerate([1, 2]):

        # instanciate VQC.
        vqc = QuanticVQC(gen_var_form=gen_two_local(reps=n_reps))

        # Get data. We will use a toy dataset here.
        X, y = get_linearly_separable_dataset(n_samples=n_samples)

        # Compute and display weight variability after training
        axe = axes[i, j]
        # ... This is all done in this method
        # It displays a spiral. Each "branch of the spiral" is a parameter inside VQC.
        # The larger is the branch, the higher is the parameter variability.
        weights_spiral(axe, vqc, X, y, n_trainings=5)
        n_params = vqc.parameter_count

        # Just improve display of the graphics. 
        if j == 0:
            axe.set_ylabel(f"n_samples: {n_samples}")
        if i == 0:
            axe.set_xlabel(f"n_params: {n_params}")
        axe.xaxis.set_label_position('top') 
        axe.set_xticks(())
        axe.set_yticks(())

plt.tight_layout()
plt.show()
