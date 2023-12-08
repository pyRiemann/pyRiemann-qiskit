"""
Contains helper methods and classes to manage datasets.
"""
from warnings import warn
import numpy as np

try:
    from mne import io, read_events, pick_types, Epochs
    from mne.datasets import sample
except Exception:
    warn("mne not available. get_mne_sample will fail.")
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.datasets import make_classification


def get_mne_sample(n_trials=10, include_auditory=False):
    """Return sample data from the MNE dataset.

    ``
    In this experiment, checkerboard patterns were presented to the subject
    into the left and right visual field, interspersed by tones to the
    left or right ear. The interval between the stimuli was 750 ms.
    Occasionally a smiley face was presented at the center of the visual field.
    The subject was asked to press a key with the right index finger
    as soon as possible after the appearance of the face.
    `` [1]_

    The samples returned by this method are epochs of duration 1s.
    Only visual left and right trials are selected.
    Data are returned filtered.

    Parameters
    ----------
    n_trials : int (default:10)
        Number of trials to return.
        If -1, then all trials are returned.
    include_auditory : boolean (default:False)
        If True, it returns also the auditory stimulation in the MNE dataset.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        ndarray of trials.
    y : ndarray, shape (n_trials,)
        Predicted target vector relative to X.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Possibility to include auditory stimulation.

    References
    ----------
    .. [1] Available from: \
        https://mne.tools/stable/overview/datasets_index.html

    """
    data_path = str(sample.data_path())

    # Set parameters and read data
    raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
    event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
    tmin, tmax = -0.0, 1
    if include_auditory:
        event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
    else:
        event_id = dict(vis_l=3, vis_r=4)

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


def generate_qiskit_dataset(n_samples=30):
    """Return qiskit dataset.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added `n_samples` parameter.
        Rename from `get_qiskit_dataset` to `generate_qiskit_dataset`.

    Parameters
    ----------
    n_samples : int (default: 30)
        Number of trials to return.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.

    """

    feature_dim = 2
    X, _, _, _ = ad_hoc_data(
        training_size=n_samples, test_size=0, n=feature_dim, gap=0.3, plot_data=False
    )

    y = np.concatenate(([0] * n_samples, [1] * n_samples))

    return (X, y)


def generate_linearly_separable_dataset(n_samples=100):
    """Return a linearly separable dataset.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added `n_samples` parameter.
        Rename from `get_linearly_separable_dataset` to
        `generate_linearly_separable_dataset`.

    Parameters
    ----------
    n_samples : int (default: 100)
        Number of trials to return.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y: ndarray, shape (n_samples,)
        Predicted target vector relative to X.

    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    return (X, y)


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

    Notes
    -----
    .. versionadded:: 0.0.2

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


class MockDataset:
    """A dataset with mock data.

    Parameters
    ----------
    dataset_gen : Callable[[], (List, List)]
        A function to generate datasets.
        The function accepts no parameters and returns a pair of lists.
        The first list contains the samples, the second one the labels.
    n_subjects : int
        The number of subjects in the dataset.
        A dataset will be generated for all subjects using the handler
        `dataset_gen`.

    Attributes
    ----------
    code_ : str
        The code of the dataset, which is also the string representation
        of the dataset.
    data_ : dict
        A dictionnary representing the dataset, e.g.:
        ``{subject1: (samples1, labels1), subject2: (samples2, labels2), ...}``
    subjects_ : list[int]
        The subjects of the dataset.

    Notes
    -----
    .. versionadded:: 0.0.3

    """

    def __init__(self, dataset_gen, n_subjects: int):
        self.code_ = "MockDataset"
        self.subjects_ = range(n_subjects)
        self.data_ = {}
        for subject in self.subjects_:
            self.data_[subject] = dataset_gen()

    def get_data(self, subject):
        """
        Returns the data of a subject.

        Parameters
        ----------
        subject : int
            A subject in the list of subjects `subjects_`.

        Returns
        -------
        data : the subject's data.

        Notes
        -----
        .. versionadded:: 0.0.3
        """
        return self.data_[subject]

    def __str__(self) -> str:
        return self.code_
