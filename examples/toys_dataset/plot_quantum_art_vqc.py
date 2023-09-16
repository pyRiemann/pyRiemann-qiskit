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

weights = []
for i in range(5):
    X, y = get_qiskit_dataset(n_trials=2)

    vqc.fit(X, y)

    train_weights = vqc._classifier.weights
    weights.append(train_weights)
    # print(vqc._classifier.ansatz.parameters)
    

df = pd.DataFrame(weights)

theta = np.arange(0, 8*np.pi, 0.1)
a = 1
b = .2

n_params = len(df)

max_var = max([df[i].var() for i in range(n_params)])

for i in range(n_params):

    dt = 2*np.pi / n_params * i
    x = a*np.cos(theta + dt)*np.exp(b*theta)
    y = a*np.sin(theta + dt)*np.exp(b*theta)
    # print(df.iloc[i])
    var = df[i].var()
    print(max_var)
    print(var)
    # dt = dt + np.pi/4.0
    
    dt = dt + (var / max_var) * np.pi/4.0

    x2 = a*np.cos(theta + dt)*np.exp(b*theta)
    y2 = a*np.sin(theta + dt)*np.exp(b*theta)

    xf = np.concatenate((x, x2[::-1]))
    yf = np.concatenate((y, y2[::-1]))

    p1 = plt.fill(xf, yf)

plt.show()