import pytest
import numpy as np
from functools import partial

from pyriemann.datasets import make_covariances


def requires_module(function, name, call=None):
    """Skip a test if package is not available (decorator)."""
    call = ("import %s" % name) if call is None else call
    reason = "Test %s skipped, requires %s." % (function.__name__, name)
    try:
        exec(call) in globals(), locals()
    except Exception as exc:
        if len(str(exc)) > 0 and str(exc) != "No module named %s" % name:
            reason += " Got exception (%s)" % (exc,)
        skip = True
    else:
        skip = False
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_matplotlib = partial(requires_module, name="matplotlib")
requires_seaborn = partial(requires_module, name="seaborn")


@pytest.fixture
def rndstate():
    return np.random.RandomState(1234)


@pytest.fixture
def get_covmats(rndstate):
    def _gen_cov(n_matrices, n_channels):
        return make_covariances(n_matrices, n_channels, rndstate,
                                return_params=False)

    return _gen_cov


@pytest.fixture
def get_covmats_params(rndstate):
    def _gen_cov_params(n_matrices, n_channels):
        return make_covariances(n_matrices, n_channels, rndstate,
                                return_params=True)

    return _gen_cov_params


def _get_labels(n_matrices, n_classes):
    return np.arange(n_classes).repeat(n_matrices // n_classes)


def _get_rand_feats(n_samples, n_features, rs):
    """Generate a set of n_features-dimensional samples for test purpose"""
    return rs.randn(n_samples, n_features)


def _get_binary_feats(n_samples, n_features):
    """Generate a balanced binary set of n_features-dimensional
     samples for test purpose, containing either 0 or 1"""
    n_classes = 2
    class_len = n_samples // n_classes  # balanced set
    samples_0 = np.zeros((class_len, n_features))
    samples_1 = np.ones((class_len, n_features))
    samples = np.concatenate((samples_0, samples_1), axis=0)
    return samples


@pytest.fixture
def prepare_data(rndstate):
    # Note: the n_classes parameters might be misleading as it is only
    # recognized by the _get_labels methods.
    def _get_dataset(n_samples, n_features, n_classes, random=True):
        if random:
            samples = _get_rand_feat(n_samples, n_features, rndstate)
        else:
            samples = _get_feats(n_samples, n_features)
        labels = _get_labels(n_samples, n_classes)
        return samples, labels
    return _prepare_data


def _get_linear_entanglement(n_qbits_in_block, n_features):
    return [list(range(i, i + n_qbits_in_block))
            for i in range(n_features - n_qbits_in_block + 1)]


def _get_pauli_z_rep_linear_entanglement(n_features):
    num_qubits_by_block = [1, 2]
    indices_by_block = []
    for n in num_qubits_by_block:
        linear = _get_linear_entanglement(n, n_features)
        indices_by_block.append(linear)
    return indices_by_block


@pytest.fixture
def get_pauli_z_linear_entangl_handle():
    def _get_pauli_z_linear_entangl_handle(n_features):
        indices = _get_pauli_z_rep_linear_entanglement(n_features)
        return lambda _: [indices]

    return _get_pauli_z_linear_entangl_handle


@pytest.fixture
def get_pauli_z_linear_entangl_idx():
    def _get_pauli_z_linear_entangl_idx(reps, n_features):
        indices = _get_pauli_z_rep_linear_entanglement(n_features)
        return [indices for _ in range(reps)]

    return _get_pauli_z_linear_entangl_idx
