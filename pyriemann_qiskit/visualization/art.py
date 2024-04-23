"""
Artistic visualization for quantum.
"""

import pandas as pd
import numpy as np


def weights_spiral(axe, vqc, X, y, n_trainings=5):
    """Artistic representation of vqc training.

    Display a spiral. Each "branch" of the spiral corresponds to a parameter inside VQC.
    When the branch is "large" it means that the weight of the parameter varies a lot
    between different trainings.

    Notes
    -----
    .. versionadded:: 0.1.0

    Parameters
    ----------
    axe: Axe
        Pointer to the matplotlib plot or subplot.
    vqc: QuanticVQC
        The instance of VQC to evaluate.
    X: ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.
    n_trainings: int (default: 5)
        Number of trainings to run, in order to evaluate the variability of the
        parameters' weights.

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.

    """

    weights = []

    for i in range(5):
        vqc.fit(X, y)
        train_weights = vqc._classifier.weights
        weights.append(train_weights)

    df = pd.DataFrame(weights)

    theta = np.arange(0, 8 * np.pi, 0.1)
    a = 1
    b = 0.2

    n_params = len(df.columns)

    max_var = df.var().max()

    # https://matplotlib.org/3.1.1/gallery/misc/fill_spiral.html
    for i in range(n_params):
        dt = 2 * np.pi / n_params * i
        x = a * np.cos(theta + dt) * np.exp(b * theta)
        y = a * np.sin(theta + dt) * np.exp(b * theta)

        var = df[i].var()

        dt = dt + (var / max_var) * np.pi / 4.0

        x2 = a * np.cos(theta + dt) * np.exp(b * theta)
        y2 = a * np.sin(theta + dt) * np.exp(b * theta)

        xf = np.concatenate((x, x2[::-1]))
        yf = np.concatenate((y, y2[::-1]))

        axe.fill(xf, yf)
