import numpy as np
from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.datasets import make_classification


def get_mne_sample(n_trials=10):
    """Return sample data from the mne dataset.

    ```
    In this experiment, checkerboard patterns were presented to the subject
    into the left and right visual field, interspersed by tones to the
    left or right ear. The interval between the stimuli was 750 ms.
    Occasionally a smiley face was presented at the center of the visual field.
    The subject was asked to press a key with the right index finger
    as soon as possible after the appearance of the face.
    ``` [1]_

    The samples returned by this method are epochs of duration 1s.
    Only visual left and right trials are selected.
    Data are returned filtered.

    Parameters
    ----------
    n_trials : int (default:10)
        Number of trials to return.
        If -1, then all trials are returned.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        ndarray of trials.
    y : ndarray, shape (n_trials,)
        Predicted target vector relative to X.

    References
    ----------
    .. [1] Available from: \
        https://mne.tools/stable/overview/datasets_index.html

    """
    data_path = sample.data_path()

    # Set parameters and read data
    raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
    event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
    tmin, tmax = -0.0, 1
    event_id = dict(vis_l=3, vis_r=4)  # select only two classes

    # Setup for reading the raw data
    raw = io.Raw(raw_fname, preload=True, verbose=False)
    raw.filter(2, None, method="iir")  # replace baselining with high-pass
    events = read_events(event_fname)

    raw.info["bads"] = ["MEG 2443"]  # set bad channels
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Read epochs
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=False,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()[:n_trials, :-1, :-1]
    y = epochs.events[:n_trials, -1]

    return X, y


def get_qiskit_dataset():
    """Return qiskit dataset.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.

    """

    feature_dim = 2
    _, inputs, _, _ = ad_hoc_data(
        training_size=30,
        test_size=0,
        n=feature_dim,
        gap=0.3,
        plot_data=False
    )

    X = np.concatenate((inputs['A'], inputs['B']))
    y = np.concatenate(([0] * 30, [1] * 30))

    return (X, y)


def get_linearly_separable_dataset():
    """Return a linearly separable dataset.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.

    """
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    return(X, y)


def get_feature_dimension(dataset):
    # This code is part of Qiskit.
    #
    # (C) Copyright IBM 2018, 2021.
    #
    # This code is licensed under the Apache License, Version 2.0. You may
    # obtain a copy of this license in the LICENSE.txt file
    # in the root directory of this source tree or at
    # http://www.apache.org/licenses/LICENSE-2.0.
    #
    # Any modifications or derivative works of this code must retain this
    # copyright notice, and modified files need to carry a notice indicating
    # that they have been altered from the originals.
    """
    Return the feature dimension of a given dataset.

    Parameters
    ----------
    dataset : dict
        key is the class name and value is the data.

    Returns
    -------
        n_features : int
            The feature dimension, -1 denotes no data in the dataset.

    Raises
    -------
    TypeError
        invalid data set

    """
    if not isinstance(dataset, dict):
        raise TypeError("Dataset is not formatted as a dict. Please check it.")

    for v in dataset.values():
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        return v.shape[1]

    return -1
