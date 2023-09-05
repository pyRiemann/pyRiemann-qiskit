"""
====================================================================
Multiclass EEG classification with Quantum Pipeline
====================================================================

This example demonstrates multiclass EEG classification with a quantum
classifier.
We will be comparing the performance of VQC vs Quantum SVM vs
Classical SVM vs Quantum MDM vs MDM.
Execution takes approximately 1h.
"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_quantum_svm
# License: BSD (3-clause)

from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
    QuantumMDMWithRiemannianPipeline,
)
from pyriemann.estimation import ERPCovariances
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


print(__doc__)

###############################################################################
# Get the data

# Use MNE sample. The include_auditory parameter select 4 classes.
X, y = get_mne_sample(n_trials=-1, include_auditory=True)

# evaluation without k-fold cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

###############################################################################
# Decoding in tangent space with a quantum classifier

# Our helper class QuantumClassifierWithDefaultRiemannianPipeline allows to
# auto-configure the parameters of the pipelines.
# Warning: these are not optimal parameters

# Piepeline 1
quantum_svm = QuantumClassifierWithDefaultRiemannianPipeline(
    dim_red=PCA(n_components=5),
)

# Piepeline 2
classical_svm = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None,  # 'None' forces classic SVM
    dim_red=PCA(n_components=5),
)

# Piepeline 3
vqc = QuantumClassifierWithDefaultRiemannianPipeline(
    dim_red=PCA(n_components=5),
    # These parameters are specific to VQC.
    # The pipeline will detect this and instantiate a VQC under the hood
    spsa_trials=40,
    two_local_reps=3,
)

# Piepeline 4
quantum_mdm = QuantumMDMWithRiemannianPipeline()

# Piepeline 5
mdm = make_pipeline(ERPCovariances(estimator="lwf"), MDM())

classifiers = [vqc, quantum_svm, classical_svm, quantum_mdm, mdm]

n_classifiers = len(classifiers)

# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side
f, axes = plt.subplots(1, n_classifiers, sharey="row")

disp = None

# Compute results
for idx in range(n_classifiers):
    # Training and classification
    clf = classifiers[idx]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Printing the results
    acc = balanced_accuracy_score(y_test, y_pred)
    acc_str = "%0.2f" % acc

    # Results visualization
    # A confusion matrix is reported for each classifier. A perfectly performing
    # classifier will have only its diagonal filled and the rest will be zeros.
    names = ["aud left", "aud right", "vis left", "vis right"]
    title = (
        ("VQC (" if idx == 0 else "Quantum SVM (" if idx == 1 else "Classical SVM (")
        if idx == 2
        else "Quantum MDM ("
        if idx == 3
        else "R-MDM (" + acc_str + ")"
    )
    axe = axes[idx]
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=axe, xticks_rotation=45)
    disp.ax_.set_title(title)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel("")
    if idx > 0:
        disp.ax_.set_ylabel("")

# Display all the confusion matrices
if disp:
    f.text(0.4, 0.1, "Predicted label", ha="left")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
