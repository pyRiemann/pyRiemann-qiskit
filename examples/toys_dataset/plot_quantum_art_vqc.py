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

vqc = QuanticVQC(gen_var_form=gen_two_local(reps=1))

X, y = get_qiskit_dataset(n_trials=2)

weights_spiral(vqc, X, y)

# 2 datasets, 2 var form, 2-3 n_trials differents