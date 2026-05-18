"""Helper methods and classes to manage datasets."""

from warnings import warn

try:
    from mne import Epochs, io, pick_types, read_events
    from mne.datasets import sample
except Exception:
    warn("mne not available. get_mne_sample will fail.")
import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.datasets import make_classification


def get_mne_sample(n_trials=10, include_auditory=False):
    """Return sample data from the MNE dataset [1]_.

    Epochs of duration 1s from a checkerboard/tone experiment.
    Only visual left and right trials are selected, returned filtered.

    Parameters
    ----------
    n_trials : int (default:10)
        Number of trials to return. If -1, all trials are returned.
    include_auditory : bool (default:False)
        If True, also return auditory stimulation trials.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Array of trials.
    y : ndarray, shape (n_trials,)
        Target vector relative to X.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added `include_auditory` parameter.
    .. versionchanged:: 0.6.0
        Moved to utils module.

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
    """Return a Qiskit ad-hoc dataset.

    Parameters
    ----------
    n_samples : int (default: 30)
        Number of samples per class.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Target vector.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added `n_samples` parameter.
        Renamed from `get_qiskit_dataset`.
    .. versionchanged:: 0.6.0
        Moved to utils module.

    """

    feature_dim = 2
    X, _, _, _ = ad_hoc_data(
        training_size=n_samples, test_size=0, n=feature_dim, gap=0.3, plot_data=False
    )

    y = np.concatenate(([0] * n_samples, [1] * n_samples))

    return (X, y)


def generate_linearly_separable_dataset(n_samples=100):
    """Return a linearly separable dataset.

    Parameters
    ----------
    n_samples : int (default: 100)
        Number of samples.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Target vector.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.1.0
        Added `n_samples` parameter.
        Renamed from `get_linearly_separable_dataset`.
    .. versionchanged:: 0.6.0
        Moved to utils module.

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
    """Return the feature dimension of a dataset.

    Based on code from Qiskit, (C) Copyright IBM 2018, 2021, Apache License 2.0.

    Parameters
    ----------
    dataset : dict
        Keys are class names, values are data arrays.

    Returns
    -------
    n_features : int
        Feature dimension, or -1 if the dataset is empty.

    Raises
    ------
    TypeError
        If `dataset` is not a dict.

    Notes
    -----
    .. versionadded:: 0.0.2
    .. versionchanged:: 0.6.0
        Moved to utils module.

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
        A dictionary representing the dataset, e.g.:
        ``{subject1: (samples1, labels1), subject2: (samples2, labels2), ...}``
    subjects_ : list[int]
        The subjects of the dataset.

    Notes
    -----
    .. versionadded:: 0.0.3
    .. versionchanged:: 0.6.0
        Moved to utils module.
    .. deprecated:: 0.7.0
        ``MockDataset`` will be removed in 0.8.0.

    """

    def __init__(self, dataset_gen, n_subjects: int):
        warn(
            "MockDataset is deprecated and will be removed in 0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.code_ = "MockDataset"
        self.subjects_ = range(n_subjects)
        self.data_ = {}
        for subject in self.subjects_:
            self.data_[subject] = dataset_gen()

    def get_data(self, subject):
        """Return the data of a subject.

        Parameters
        ----------
        subject : int
            A subject in `subjects_`.

        Returns
        -------
        data : tuple
            The subject's (samples, labels) data.

        Notes
        -----
        .. versionadded:: 0.0.3
        .. versionchanged:: 0.6.0
            Moved to utils module.

        """
        return self.data_[subject]

    def __str__(self) -> str:
        return self.code_
