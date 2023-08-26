"""
====================================================================
Multiclass EEG classification with Quantum Pipeline
====================================================================

This example demonstrate multiclass EEG classification with a quantum
classifier.
We will be comparing the performance of VQC vs Quantum SVM vs
Classical SVM
"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_quantum_svm
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticSVM
from pyriemann_qiskit.utils.filtering import NaiveDimRed
from pyriemann_qiskit.datasets import get_mne_sample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from matplotlib import pyplot as plt


print(__doc__)

###############################################################################
# Get the data

# Use MNE sample. The include_auditory parameter select 4 classes.
X, y = get_mne_sample(n_trials=-1, include_auditory=True)

# ...skipping the KFold validation parts (for the purpose of the test only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

###############################################################################
# Decoding in tangent space with a quantum classifier

# We will use the QuantumClassifierWithDefaultRiemannianPipeline handler
# To configure the parameters of the pipeline.

quantum_svm = QuantumClassifierWithDefaultRiemannianPipeline(
    dim_red=PCA(n_components=5),
)

classical_svm = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None,  # 'None' forces classic SVM
    dim_red=PCA(n_components=5),
)

vqc = QuantumClassifierWithDefaultRiemannianPipeline(
    dim_red=PCA(n_components=5),
    # These parameters are specific to VQC.
    # The pipeline will detect this and instantiate a VQC under the hood
    spsa_trials=40,
    two_local_reps=3,
)

classifiers = [vqc, quantum_svm, classical_svm]

# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side
f, axes = plt.subplots(1, 2, sharey="row")

disp = None

# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Printing the results
    acc = balanced_accuracy_score(y_pred, y_test)
    acc_str = "%0.2f" % acc

    names = ["vis left", "vis right"]
    title = ("Quantum (" if quantum else "Classical (") + acc_str + ")"
    axe = axes[0 if quantum else 1]
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=axe, xticks_rotation=45)
    disp.ax_.set_title(title)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel("")
    if not quantum:
        disp.ax_.set_ylabel("")

if disp:
    f.text(0.4, 0.1, "Predicted label", ha="left")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
