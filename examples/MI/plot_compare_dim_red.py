"""
====================================================================
Determine best dimension reduction technics
====================================================================

Fine tune quantum pipeline, by determining the best dimension
reduction technics for diminishing the size of the feature vectors.

To achieve this goal, we used a sample set from MNE and
GridSearchCV from the sklearn library.

Although only a limited number of parameter combination was used in
this example (in order to limit computation time), a similar approach
can be used to fine-tuned other hyper parameters such as the feature
entanglement or the number of shots.

"""

# Author: Gregoire Cattan
# License: BSD (3-clause)

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pyriemann_qiskit.datasets import get_mne_sample
from pyriemann_qiskit.pipelines import QuantumClassifierWithDefaultRiemannianPipeline
from pyriemann_qiskit.utils.filtering import NaiveDimRed

print(__doc__)

###############################################################################

X, y = get_mne_sample(n_trials=10)

default_params = {
    # size of the xdawn filter
    "nfilter": [2],  # [1, 2, 3]
    # hyperparameter for the SVC classifier
    "gamma": [0.1],  # [None, 0.05, 0.1, 0.15]
    # Determine the number of "run" on the quantum machine (simulated or real)
    # the higher is this number, the lower the variability.
    "shots": [1024],  # [512, 1024, 2048]
    # This parameter changes the depth of the circuit when entangling data.
    # There is a trade-off between accuracy and noise when the depth of the
    # circuit increases.
    "feature_reps": [2],  # [2, 3, 4]
    # These parameters set-up the optimizer when using VQC.
    "spsa_trials": [None],  # [40]
    "two_local_reps": [None],  # [2, 3, 4]
    # After the data are projected into the tangentspace,
    # we can reduce the size of the resulting vector.
    # Computational time tends to increases with the dimension of the feature,
    # especially when using a simulated quantum machine.
    # A quantum simulator is also limited to only 24qbits
    # (and so is the size of the feature).
    "dim_red": [PCA(n_components=10), NaiveDimRed()],
}

pipe = QuantumClassifierWithDefaultRiemannianPipeline()


def customize(custom_params: dict):
    new_params = {}
    for key in default_params:
        new_params[key] = (
            custom_params[key] if key in custom_params else default_params[key]
        )
    return new_params


def search(params: dict):
    grid = GridSearchCV(
        pipe,
        params,
        scoring="balanced_accuracy",
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=2),
    )
    grid.fit(X, y)
    return grid.best_params_


def analyze_multiple(l_params: list):
    best_params = []
    for params in l_params:
        params_space = customize(params)
        best_params.append(search(params_space))
    print(best_params)


SVC = {"shots": [None]}
QSVC = {}
VQC = {"spsa_trials": [40], "two_local_reps": [2]}

analyze_multiple([SVC, QSVC, VQC])
