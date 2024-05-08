"""Module containing helpers for IBM quantum backends
   providers and simulators."""

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from qiskit_machine_learning.kernels import (
    FidelityStatevectorKernel,
    FidelityQuantumKernel,
)
from qiskit_algorithms.state_fidelities import ComputeUncompute
import logging


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


def get_quantum_kernel(feature_map, quantum_instance, use_fidelity_state_vector_kernel):
    """Get a quantum kernel

    Return an instance of FidelityQuantumKernel or
    FidelityStatevectorKernel (in the case of a simulation).

    Parameters
    ----------
    feature_map: QuantumCircuit | FeatureMap
        An instance of a feature map.
    quantum_instance: BaseSampler
        A instance of BaseSampler.
    use_fidelity_state_vector_kernel: boolean
        if True, use a FidelitystatevectorKernel for simulation.

    Returns
    -------
    kernel: QuantumKernel
        The quantum kernel.

    Notes
    -----
    .. versionadded:: 0.3.0
    """
    if use_fidelity_state_vector_kernel and isinstance(
        quantum_instance._backend, AerSimulator
    ):
        logging.log(
            logging.WARN,
            """FidelityQuantumKernel skipped because of time.
                    Using FidelityStatevectorKernel with AerStatevector.
                    Seed cannot be set with FidelityStatevectorKernel.
                    Increase the number of shots to diminish the noise.""",
        )

        # if this is a simulation,
        # we will not use FidelityQuantumKernel as it is slow. See
        # https://github.com/qiskit-community/qiskit-machine-learning/issues/547#issuecomment-1486527297
        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
            statevector_type=AerStatevector,
            shots=quantum_instance.options["shots"],
        )
    else:
        kernel = FidelityQuantumKernel(
            feature_map=feature_map, fidelity=ComputeUncompute(quantum_instance)
        )
    return kernel
