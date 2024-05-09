"""
====================================================================
Plot training curve of VQC and QAOA
====================================================================

Plot the training curve of quantum neural network VQC and QAOA.

"""

# Author: Gregoire Cattan
# License: BSD (3-clause)

from pyriemann_qiskit.datasets.utils import get_mne_sample
from pyriemann_qiskit.pipelines import QuantumMDMWithRiemannianPipeline
from pyriemann_qiskit.utils.hyper_params_factory import gen_two_local, get_spsa
import matplotlib.pyplot as plt
from pyriemann_qiskit.datasets import generate_linearly_separable_dataset
from pyriemann_qiskit.classification import QuanticNCH, QuanticVQC
from pyriemann_qiskit.visualization import weights_spiral
from pyriemann.estimation import XdawnCovariances, Shrinkage

print(__doc__)

###############################################################################
# In this example we will display weights variability of the parameter inside
# the variational quantum circuit which is used by VQC.
#
# The idea is simple:
#
# - We initialize a VQC with different number of parameters and number of samples.
# - We train the VQC a couple of times and we store the fitted weights.
# - We compute variability of the weight and display it in a fashionable way.

# Let's start by defining some plot area.
fig, axes = plt.subplots(1, 2)
fig.suptitle("Training curve")

X, y = generate_linearly_separable_dataset(n_samples=20)
vqc = QuanticVQC()
vqc.fit(X, y)

evaluated_values = vqc.evaluated_values_
print(evaluated_values)
axe = axes[0]
axe.plot(evaluated_values)
axe.set_ylabel("Evaluated values (VQC)")
axe.set_xlabel("Evaluations")

X, y = get_mne_sample(n_trials=100)
mdm = QuantumMDMWithRiemannianPipeline(
    metric={"mean": "qeuclid", "distance": "euclid"},
    quantum=True,
    regularization=Shrinkage(shrinkage=0.9),
    shots=1024,
    seed=696288,
    qaoa_optimizer=get_spsa()
)
mdm.fit(X, y)
evaluated_values = mdm._pipe[2]._optimizer.evaluated_values_
print(evaluated_values)
axe = axes[1]
axe.plot(evaluated_values)
axe.set_ylabel("Evaluated values (QAOA)")
axe.set_xlabel("Evaluations")

# axe.set_xticks(())
# axe.set_yticks(())

plt.tight_layout()
plt.show()
