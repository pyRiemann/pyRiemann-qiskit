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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.svm import SVC
from pyriemann_qiskit.datasets import get_linearly_separable_dataset, get_qiskit_dataset
from pyriemann_qiskit.classification import QuanticVQC

# uncomment to run comparison with QuanticVQC (disabled for CI/CD)
# from pyriemann_qiskit.classification import QuanticVQC


print(__doc__)

###############################################################################

vqc = QuanticVQC()

for i in range(10):
    X, y = get_qiskit_dataset(n_trials=2)

    vqc.fit(X, y)

    train_weights = vqc._classifier.weights

    sns.lineplot(train_weights)



plt.show()


