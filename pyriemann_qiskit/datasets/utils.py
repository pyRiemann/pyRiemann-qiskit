import numpy as np
from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample
from qiskit.ml.datasets import ad_hoc_data
from sklearn.datasets import make_classification


def get_mne_sample(samples=10):

    data_path = sample.data_path()

    ###############################################################################
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

    X = epochs.get_data()[:10]
    y = epochs.events[:, -1][:10]

    return X, y


def get_qiskit_dataset():
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
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    return(X, y)