import numpy as np
from mne import io, read_events, pick_types, Epochs, find_events
from mne.datasets import sample
from qiskit.ml.datasets import ad_hoc_data
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from braininvaders2012.dataset import BrainInvaders2012


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


def get_bi2012_dataset(training=True,
                       fmin=1, fmax=24, sampling=128, tmin=0.1, tmax=0.7,
                       baseline=None, verbose=False):
    """Return sample data from the brain invaders 2012 dataset.

    Return an iterator over the brain invaders 2012 dataset [1]_.
    Each call to the _next_ method return the epochs `X` and the predicted
    target vector `y` relative to `X` for a subject in the dataset.

    The dataset contains the EEG recording of 26 participants playing a BCI
    version of the vintage game Brain Invaders, based on the
    oddball paradigm (P300).
    Each subject participated in a Training and Online session of the game.

    A detailed description of the dataset can be found in [2]_.

    Parameters
    ----------
    training : bool (default:True)
        If true will only download and analyze data related
        to the training session.
    fmin : int (default:1)
        Minimum frequence (Hz) for passband filtering (data preprocessing).
    fmax : int (default:24)
        Maximum frequence (Hz) for passband filtering (data preprocessing).
    sampling : int (default:128)
        Data were initially sampled at 128Hz. You can resample the data using
        this parameter.
    tmin : int (default:0.1)
        Start time before event.
    tmax : int (default:0.7)
        Stop time before event.
    baseline : None | tuple[2] (default:None)
        The time interval to apply baseline correction [3]_.
    verbose : bool (default:False)
        Print all traces if true.

    Returns
    -------
    dataset: Iterator
       An iterator over the brain invaders 2012 dataset.

    References
    ----------
    .. [1] Available from: \
        https://github.com/plcrodrigues/py.BI.EEG.2012-GIPSA
    .. [2] G. F. P. Van Veen et al. \
        ‘Building Brain Invaders: EEG data of an experimental validation’
        (mai 2019). Available at:  \
        https://hal.archives-ouvertes.fr/hal-02126068
    .. [3] \
        https://mne.tools/0.20/generated/mne.Epochs.html#mne.Epochs

    """

    dataset = BrainInvaders2012(Training=training, Online=not training)
    subjects = dataset.subject_list

    class Iterator():
        def _iter_(self):
            self.index = 0

        def _next_(self):
            if self.index == len(subjects - 1):
                raise StopIteration

            subject = subjects[self.index]
            self.index += 1
            data = dataset._get_single_subject_data(subject)
            run = "run_" + ("training" if training else "online")
            raw = data['session_1'][run]

            # filter data and resample
            raw.filter(fmin, fmax, verbose=verbose)
            raw.resample(sampling)

            # detect the events and cut the signal into epochs
            events = find_events(raw=raw, shortest_event=1, verbose=verbose)
            event_id = {'NonTarget': 1, 'Target': 2}
            epochs = Epochs(raw, events, event_id, tmin, tmax,
                            baseline=baseline, verbose=verbose, preload=True)
            epochs.pick_types(eeg=True)

            # get trials and labels
            X = epochs.get_data()
            y = events[:, -1]
            y = LabelEncoder().fit_transform(y)
            return (X, y)

    return Iterator()
