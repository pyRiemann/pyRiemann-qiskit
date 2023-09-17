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

import pandas as pd
import numpy as np
from pyriemann_qiskit.utils.hyper_params_factory import gen_two_local
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.svm import SVC
from pyriemann_qiskit.datasets import get_linearly_separable_dataset, get_qiskit_dataset
from pyriemann_qiskit.classification import QuanticVQC
from pyriemann_qiskit.visualization import weights_spiral


print(__doc__)

###############################################################################

fig, axes = plt.subplots(2, 3)
fig.suptitle('Vertically stacked subplots')

vqc_low_param = QuanticVQC(gen_var_form=gen_two_local(reps=1))
vqc = QuanticVQC(gen_var_form=gen_two_local(reps=2))

X, y = get_qiskit_dataset(n_samples=2)
X2, y2 = get_qiskit_dataset(n_samples=30)

weights_spiral(axes[0, 0], vqc, X, y)
axes[0, 0].set_title("High param, qiskit dataset, low samples")
weights_spiral(axes[0, 1], vqc_low_param, X, y)
axes[0, 1].set_title("Low param, qiskit dataset, low samples")
weights_spiral(axes[0, 2], vqc, X2, y2)
axes[0, 2].set_title("High param, qiskit dataset, high samples")


X, y = get_linearly_separable_dataset(n_samples=2)
X2, y2 = get_linearly_separable_dataset(n_samples=30)

weights_spiral(axes[1, 0], vqc, X, y)
axes[1, 0].set_title("High param, linear dataset, low samples")
weights_spiral(axes[1, 1], vqc_low_param, X, y)
axes[1, 1].set_title("Low param, linear dataset, low samples")
weights_spiral(axes[1, 2], vqc, X2, y2)
axes[1, 2].set_title("High param, linear dataset, high samples")


plt.show()
# 2 datasets, 2 var form, 2-3 n_trials differents