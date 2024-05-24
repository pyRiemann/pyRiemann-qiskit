import logging

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.base import TransformerMixin


def _ansatz(num_qubits):
    return RealAmplitudes(num_qubits, reps=5)


def _auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(
        _ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True
    )
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash

    # swap test
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit


class BasicQnnAutoencoder(TransformerMixin):

    """Quantum denoising

    This class implements a quantum auto encoder.
    The implementation was adapted from [1]_, to be compatible with scikit-learn.

    Parameters
    ----------
    num_latent : int, default=3
        The number of qubits in the latent space.
    num_trash : int, default=2
        The number of qubits in the trash space.
    opt : Optimizer, default=SPSA(maxiter=100, blocking=True)
        The classical optimizer to use.
    callback : Callable[int, double], default=None
        An additional callback for the optimizer.
        The first parameter is the number of cost evaluation call.
        The second parameter is the cost.

    Notes
    -----
    .. versionadded:: 0.3.0

    Attributes
    ----------
    costs_ : list
        The values of the cost function.
    fidelities_ : list, shape (n_samples,)
        fidelities (one fidelity for each sample).

    References
    ----------
    .. [1] \
        https://qiskit-community.github.io/qiskit-machine-learning/tutorials/12_quantum_autoencoder.html
    .. [2] A. Mostafa et al., 2024 \
        ‘Quantum Denoising in the Realm of Brain-Computer Interfaces:
        A Preliminary Study’,
        https://hal.science/hal-04501908


    """

    def __init__(
        self,
        num_latent=3,
        num_trash=2,
        opt=SPSA(maxiter=100, blocking=True),
        callback=None,
    ):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.opt = opt
        self.callback = callback

    def _log(self, msg):
        logging.info(f"[BasicQnnAutoencoder] {msg}")

    def _get_transformer(self):
        # encoder
        transformer = QuantumCircuit(self.n_qubits)
        transformer = transformer.compose(self._feature_map)
        ansatz_qc = _ansatz(self.n_qubits)
        transformer = transformer.compose(ansatz_qc)
        transformer.barrier()

        # trash space
        for i in range(self.num_trash):
            transformer.reset(self.num_latent + i)
        transformer.barrier()

        # decoder
        transformer = transformer.compose(ansatz_qc.inverse())
        self._transformer = transformer
        return transformer

    def _compute_fidelities(self, X):
        fidelities = []
        for x in X:
            param_values = np.concatenate((x, self._opt_result.x))
            output_qc = self._transformer.assign_parameters(param_values)
            output_state = Statevector(output_qc).data

            original_qc = self._feature_map.assign_parameters(x)
            original_state = Statevector(original_qc).data

            fidelity = np.sqrt(np.dot(original_state.conj(), output_state) ** 2)
            fidelities.append(fidelity.real)
        return fidelities

    @property
    def n_qubits(self):
        return self.num_latent + self.num_trash

    def fit(self, X, _y=None, **kwargs):
        """Fit the autoencoder.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Set of time epochs.
            n_features must equal 2 ** n_qubits,
            where n_qubits = num_trash + num_latent.

        Returns
        -------
        self : BasicQnnAutoencoder
            The BasicQnnAutoencoder instance.
        """

        _, n_features = X.shape

        self.costs_ = []
        self.fidelities_ = []
        self._iter = 0

        self._log(
            f"raw feature size: {2 ** self.n_qubits} and feature size: {n_features}"
        )
        assert 2**self.n_qubits == n_features

        self._feature_map = RawFeatureVector(2**self.n_qubits)

        self._auto_encoder = _auto_encoder_circuit(self.num_latent, self.num_trash)

        qc = QuantumCircuit(self.num_latent + 2 * self.num_trash + 1, 1)
        qc = qc.compose(self._feature_map, range(self.n_qubits))
        qc = qc.compose(self._auto_encoder)

        qnn = SamplerQNN(
            circuit=qc,
            input_params=self._feature_map.parameters,
            weight_params=self._auto_encoder.parameters,
            interpret=lambda x: x,
            output_shape=2,
        )

        def cost_func(params_values):
            self._iter += 1
            if self._iter % 10 == 0:
                self._log(f"Iteration {self._iter}")

            probabilities = qnn.forward(X, params_values)
            cost = np.sum(probabilities[:, 1]) / X.shape[0]
            self.costs_.append(cost)
            if self.callback:
                self.callback(self._iter, cost)
            return cost

        initial_point = algorithm_globals.random.random(
            self._auto_encoder.num_parameters
        )
        self._opt_result = self.opt.minimize(fun=cost_func, x0=initial_point)

        # encoder/decoder circuit
        self._transformer = self._get_transformer()

        # compute fidelity
        self.fidelities_ = self._compute_fidelities(X)

        self._log(f"Mean fidelity: {np.mean(self.fidelities_)}")

        return self

    def transform(self, X, **kwargs):
        """Apply the transformer circuit.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Set of time epochs.
            n_features must equal 2 ** n_qubits,
            where n_qubits = num_trash + num_latent.

        Returns
        -------
        outputs : ndarray, shape (n_samples, 2 ** n_qubits)
            The autocoded sample. n_qubits = num_trash + num_latent.
        """

        _, n_features = X.shape
        outputs = []
        for x in X:
            param_values = np.concatenate((x, self._opt_result.x))
            output_qc = self._transformer.assign_parameters(param_values)
            output_sv = Statevector(output_qc).data
            output_sv = np.reshape(np.abs(output_sv) ** 2, n_features)
            outputs.append(output_sv)
        return np.array(outputs)
