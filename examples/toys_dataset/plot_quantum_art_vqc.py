"""
====================================================================
Comparison with toys datasets.
====================================================================

Comparison of classification using quantum versus classical SVM
classifiers on toys datasets.

"""
# Code source:
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Modified for pyRiemann-qiskit by Gregoire Cattan
# License: BSD 3 clause

from pyriemann_qiskit.utils.hyper_params_factory import gen_two_local
import matplotlib.pyplot as plt
from pyriemann_qiskit.datasets import get_linearly_separable_dataset
from pyriemann_qiskit.classification import QuanticVQC
from pyriemann_qiskit.visualization import weights_spiral


print(__doc__)

###############################################################################

fig, axes = plt.subplots(2, 2)
fig.suptitle('Vertically stacked subplots')

vqc_low_param = QuanticVQC(gen_var_form=gen_two_local(reps=1))
vqc = QuanticVQC(gen_var_form=gen_two_local(reps=2))

for i, n_samples in enumerate([2, 3]):
    for j, n_reps in enumerate([1, 2]):
        vqc = QuanticVQC(gen_var_form=gen_two_local(reps=n_reps))
        X, y = get_linearly_separable_dataset(n_samples=n_samples)
        axe = axes[i, j]
        
        weights_spiral(axe, vqc, X, y)
        n_params = vqc.parameter_count
        if j == 0:
            axe.set_ylabel(f"n_samples: {n_samples}")
        if i == 0:
            axe.set_xlabel(f"n_params: {n_params}")
        axe.xaxis.set_label_position('top') 
        axe.set_xticks(())
        axe.set_yticks(())

plt.tight_layout()
plt.show()
