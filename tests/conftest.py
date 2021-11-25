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


@pytest.fixture
def get_labels():
    def _get_labels(n_matrices, n_classes):
        return np.arange(n_classes).repeat(n_matrices // n_classes)

    return _get_labels


def generate_feat(n_samples, n_features, rs):
    """Generate a set of n_features-dimensional samples for test purpose"""
    return rs.randn(n_samples, n_features)


@pytest.fixture
def get_feats(rndstate):
    def _gen_feat(n_samples, n_features):
        return generate_feat(n_samples, n_features, rndstate)
    return _gen_feat


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
