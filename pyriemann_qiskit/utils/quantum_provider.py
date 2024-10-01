"""Module containing helpers for IBM quantum backends
   providers and simulators."""

import joblib
import logging
import os
import pickle

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.kernels import (
    FidelityStatevectorKernel,
    FidelityQuantumKernel,
)

try:
    from qiskit_symb.quantum_info import Statevector

    QISKIT_SYMB = True
except ImportError:
    QISKIT_SYMB = False


class SymbFidelityStatevectorKernel:

    """Symbolic Statevector kernel

    An implementation of the quantum kernel for classically simulated
    state vectors [1]_ using qiskit-symb for symbolic representation
    of statevectors [2]_.

    Here, the kernel function is defined as the overlap of two simulated quantum
    statevectors produced by a parametrized quantum circuit (called feature map) [1]_.

    Notes
    -----
    .. versionadded:: 0.4.0

    Parameters
    ----------
    feature_map: QuantumCircuit | FeatureMap
        An instance of a feature map.
    gen_feature_map: Callable[[int, str], QuantumCircuit | FeatureMap], \
                      default=Callable[int, ZZFeatureMap]
        Function generating a feature map to encode data into a quantum state.
    n_jobs: int
        The number of job for fidelity evaluation.
        If null or negative, the number of jobs is set to 1
        If set to 1, evaluation will run on the main thread.

    References
    ----------
    .. [1] \
    https://github.com/qiskit-community/qiskit-machine-learning/blob/30dad803e9457f955464220eddc1e55a65452bbc/qiskit_machine_learning/kernels/fidelity_statevector_kernel.py#L31
    .. [2] https://github.com/SimoneGasperini/qiskit-symb/issues/6

    """

    def __init__(self, feature_map, gen_feature_map, n_jobs=1):
        self.n_jobs = n_jobs if n_jobs >= 1 else 1
        cached_file = os.path.join(
            "symb_statevectors", f"{feature_map.name}-{feature_map.reps}"
        )

        if os.path.isfile(cached_file):
            print("Loading symbolic Statevector from cache")
            file = open(cached_file, "rb")
            sv = pickle.load(file)
        else:
            print("Computing symbolic Statevector")
            fm2 = gen_feature_map(feature_map.num_qubits, "b")
            self.circuit = feature_map.compose(fm2.inverse()).decompose()
            sv = Statevector(self.circuit)
            print(f"Dumping to {cached_file}")
            file = open(cached_file, "wb")
            pickle.dump(sv, file)

        self.function = sv.to_lambda()

    def evaluate(self, x_vec, y_vec=None):
        """Evaluate the quantum kernel.

        Returns
        -------
        kernel : ndarray, shape (len(x_vec), len(y_vec))
            The kernel matrix.

        Notes
        -----
        .. versionadded:: 0.4.0
        """
        if y_vec is None:
            y_vec = x_vec

        x_vec_len = len(x_vec)
        y_vec_len = len(y_vec)

        is_sim = x_vec_len == y_vec_len and (x_vec == y_vec).all()

        kernel_matrix = np.zeros((x_vec_len, y_vec_len))

        chunck = x_vec_len // self.n_jobs

        def compute_fidelity_partial_matrix(i_thread):
            for i in range(i_thread * chunck, (i_thread + 1) * chunck):
                x = x_vec[i]
                for j in range(i if is_sim else y_vec_len):
                    y = y_vec[j]
                    if isinstance(x, np.float64):
                        # Pegagos implementation
                        fidelity = abs(self.function(x, y)[0, 0]) ** 2
                    else:
                        fidelity = abs(self.function(*x, *y)[0, 0]) ** 2

                    kernel_matrix[i, j] = fidelity
                    if is_sim:
                        kernel_matrix[j, i] = fidelity
            return kernel_matrix

        if self.n_jobs == 1:
            return compute_fidelity_partial_matrix(0)
        else:
            print("n_jobs greater than 1, parallelizing")
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(compute_fidelity_partial_matrix)(i_thread)
                for i_thread in range(self.n_jobs)
            )
            for result in results:
                kernel_matrix += result
            return kernel_matrix


def get_provider():
    """Return an IBM quantum provider.

    Returns
    -------
    provider : QiskitRuntimeService
        An instance of QiskitRuntimeService.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.1.0
        IBMProvider is not a static API anymore but need to be instanciated.
    .. versionchanged:: 0.3.0
        Switch from IBMProvider to QiskitRuntimeService.
    """
    return QiskitRuntimeService(channel="ibm_quantum")


def get_simulator():
    """Return a quantum simulator.

    Return a quantum simulator,
    supporting GPU and NVIDIA's cuQuantum optimization
    (if enabled).

    Returns
    -------
    simulator : AerSimulator
        A quantum simulator.

    Notes
    -----
    .. versionadded:: 0.0.4
    """
    backend = AerSimulator(method="statevector", cuStateVec_enable=True)
    if "GPU" in backend.available_devices():
        backend.set_options(device="GPU")
    else:
        print("GPU optimization disabled. No device found.")
    return backend


def get_device(provider, min_qubits):
    """Returns all real remote quantum backends.

    Returns all real remote quantum backends,
    available with the account token and having at least
    `min_qubits`.

    Parameters
    ----------
    provider: IBMProvider
        An instance of IBMProvider.
    min_qubits: int
        The minimun of qubits.

    Returns
    -------
    devices: IBMQBackend[]
        A list of compatible backends.

    Raises
    -------
    ValueError
        if no devices found.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.3.0
        Rename get_devices to get_device
        Switch from IBMProvider to QiskitRuntimeService
    """

    return provider.least_busy(
        operational=True, simulator=False, min_num_qubits=min_qubits
    )


def get_quantum_kernel(
    feature_map,
    gen_feature_map,
    quantum_instance,
    use_fidelity_state_vector_kernel,
    use_qiskit_symb,
    n_jobs=4
):
    """Get a quantum kernel

    Return an instance of FidelityQuantumKernel or
    FidelityStatevectorKernel (in the case of a simulation).

    For simulation with a small number of qubits (< 9), and `use_qiskit_symb` is True,
    qiskit-symb is used.

    Parameters
    ----------
    feature_map: QuantumCircuit | FeatureMap
        An instance of a feature map.
    quantum_instance: BaseSampler
        A instance of BaseSampler.
    use_fidelity_state_vector_kernel: boolean
        If True, use a FidelitystatevectorKernel for simulation.
    use_qiskit_symb: boolean
        This flag is used only if qiskit-symb is installed.
        If True and the number of qubits < 9, then qiskit_symb is used.
    n_jobs: boolean
        The number of jobs for the qiskit-symb fidelity state vector
        (if applicable)

    Returns
    -------
    kernel: QuantumKernel
        The quantum kernel.

    See also
    --------
    SymbFidelityStatevectorKernel

    Notes
    -----
    .. versionadded:: 0.3.0
    .. versionchanged:: 0.4.0
        Add support for qiskit-symb
    """
    if use_fidelity_state_vector_kernel and isinstance(
        quantum_instance._backend, AerSimulator
    ):
        # For simulation:
        if QISKIT_SYMB and feature_map.num_qubits <= 9 and use_qiskit_symb:
            # With a small number of qubits, let's use qiskit-symb
            # See:
            # https://medium.com/qiskit/qiskit-symb-a-qiskit-ecosystem-package-for-symbolic-quantum-computation-b6b4407fa705
            kernel = SymbFidelityStatevectorKernel(
                feature_map, gen_feature_map, n_jobs=n_jobs
            )
            logging.log(
                logging.WARN,
                """Using SymbFidelityStatevectorKernel""",
            )
        else:
            # For a larger number of qubits,
            # we will not use FidelityQuantumKernel as it is slow. See
            # https://github.com/qiskit-community/qiskit-machine-learning/issues/547#issuecomment-1486527297
            kernel = FidelityStatevectorKernel(
                feature_map=feature_map,
                statevector_type=AerStatevector,
                shots=quantum_instance.options["shots"],
            )
            logging.log(
                logging.WARN,
                """FidelityQuantumKernel skipped because of time.
                    Using FidelityStatevectorKernel with AerStatevector.
                    Seed cannot be set with FidelityStatevectorKernel.
                    Increase the number of shots to diminish the noise.""",
            )
    else:
        kernel = FidelityQuantumKernel(
            feature_map=feature_map, fidelity=ComputeUncompute(quantum_instance)
        )
    return kernel
