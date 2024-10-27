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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann_qiskit.classification import (  # uncomment to run comparison with QuanticVQC (disabled for CI/CD); QuanticVQC
    QuanticSVM,
)
from pyriemann_qiskit.datasets import (
    generate_linearly_separable_dataset,
    generate_qiskit_dataset,
)

# uncomment to run comparison with QuanticVQC (disabled for CI/CD)
# from pyriemann_qiskit.classification import QuanticVQC


print(__doc__)

###############################################################################


h = 0.02  # step size in the mesh
labels = (0, 1)
names = [
    "Linear SVM",
    "RBF SVM",
    # uncomment to run comparison with QuanticVQC (disabled for CI/CD)
    # "VQC",
    "QSVM",
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma="auto", C=0.001),
    # uncomment to run comparison with QuanticVQC (disabled for CI/CD)
    # QuanticVQC(),
    QuanticSVM(quantum=False),  # quantum=False for CI
]

# Warning: There is a known convergence issue with QSVM
# and some python versions:
# https://github.com/Qiskit/qiskit-aqua/issues/1106
# https://github.com/Qiskit/qiskit-aqua/pull/1190

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    generate_linearly_separable_dataset(),
    generate_qiskit_dataset(),
]

figure = plt.figure(figsize=(15, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
