"""
Contains helper methods and classes to manage the tests.
"""
import pytest
import numpy as np
from functools import partial
from pyriemann.datasets import make_matrices
from pyriemann_qiskit.datasets import get_mne_sample
from operator import itemgetter


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
        return make_matrices(
            n_matrices, n_channels, "spd", rndstate, return_params=False
        )

    return _gen_cov


@pytest.fixture
def get_covmats_params(rndstate):
    def _gen_cov_params(n_matrices, n_channels):
        return make_matrices(
            n_matrices, n_channels, "spd", rndstate, return_params=True
        )

    return _gen_cov_params


def _get_labels(n_matrices, n_classes):
    # Warning: this implementation only works
    # if n_classes is a divider of n_matrices.
    return np.arange(n_classes).repeat(n_matrices // n_classes)


@pytest.fixture
def get_labels():
    return _get_labels


def _get_rand_feats(n_samples, n_features, rs):
    """Generate a set of n_features-dimensional samples for test purpose"""
    return rs.randn(n_samples, n_features)


def _get_separable_feats(n_samples, n_features, n_classes):
    """Generate a balanced set of n_features-dimensional
    samples for test purpose, containing 0, 1, 2, ... n_classes as values"""
    class_len = n_samples // n_classes  # balanced set
    samples = []
    for i in range(n_classes):
        samples.append(np.zeros((class_len, n_features)) + i)
    samples = np.concatenate(samples, axis=0)
    return samples


@pytest.fixture
def get_dataset(rndstate):
    """Return a dataset with shape (n_samples, n_features).
    The dataset can contains random (type="rand") or
    binary (type="bin") values.
    If the attribute `type` is None, the default mne dataset
    is returned.
    """

    # recognized by the _get_labels methods.
    def _get_dataset(n_samples, n_features, n_classes, type="bin"):
        if type == "rand":
            samples = _get_rand_feats(n_samples, n_features, rndstate)
            labels = _get_labels(n_samples, n_classes)
        elif type == "bin":
            samples = _get_separable_feats(n_samples, n_features, n_classes)
            labels = _get_labels(n_samples, n_classes)
        elif type == "rand_cov":
            samples = make_matrices(
                n_samples, n_features, "spd", 0, return_params=False
            )
            labels = _get_labels(n_samples, n_classes)
        elif type == "bin_cov":
            samples_0 = make_matrices(
                n_samples // n_classes, n_features, "spd", 0, return_params=False
            )
            samples = np.concatenate(
                [samples_0 * (i + 1) for i in range(n_classes)], axis=0
            )
            labels = _get_labels(n_samples, n_classes)
        else:
            samples, labels = get_mne_sample()
        return samples, labels

    return _get_dataset


def _get_linear_entanglement(n_qbits_in_block, n_features):
    return [
        list(range(i, i + n_qbits_in_block))
        for i in range(n_features - n_qbits_in_block + 1)
    ]


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


class BinaryTest:
    def prepare(self, n_samples, n_features, quantum_instance, type):
        self.n_classes = 2
        self.n_samples = n_samples
        self.n_features = n_features
        self.quantum_instance = quantum_instance
        self.type = type
        self.class_len = n_samples // self.n_classes

    def test(self, get_dataset):
        # there is no __init__ method with pytest
        n_samples, n_features, quantum_instance, type = itemgetter(
            "n_samples", "n_features", "quantum_instance", "type"
        )(self.get_params())
        self.prepare(n_samples, n_features, quantum_instance, type)
        self.samples, self.labels = get_dataset(
            self.n_samples, self.n_features, self.n_classes, self.type
        )
        self.additional_steps()
        self.check()

    def get_params(self):
        raise NotImplementedError

    def additional_steps(self):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError


class BinaryFVT(BinaryTest):
    def additional_steps(self):
        self.quantum_instance.fit(self.samples, self.labels)
        self.prediction = self.quantum_instance.predict(self.samples)
        self.predict_proab = self.quantum_instance.predict_proba(self.samples)
        print(self.labels, self.prediction)


class MultiClassTest(BinaryTest):
    def prepare(self, n_samples, n_features, quantum_instance, type):
        super().prepare(n_samples, n_features, quantum_instance, type)
        self.n_classes = 3
        self.class_len = n_samples // self.n_classes


class MultiClassFVT(MultiClassTest):
    def additional_steps(self):
        BinaryFVT.additional_steps(self)
