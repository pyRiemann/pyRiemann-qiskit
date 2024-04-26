"""Module containing helpers for IBM quantum backends
   providers and simulators."""

from qiskit_ibm_provider import IBMProvider
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
    provider : IBMProvider
        An instance of IBMProvider.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.1.0
        IBMProvider is not a static API anymore but need to be instanciated.
    """
    return IBMProvider()


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


def get_devices(provider, min_qubits):
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
    """

    def filters(device):
        return (
            device.configuration().n_qubits >= min_qubits
            and not device.configuration().simulator
            and device.status().operational
        )

    devices = provider.backends(filters=filters)
    if devices is None or len(devices) == 0:
        raise ValueError(
            "No devices matching: real quantum backend, operational and n_qubits>="
            + min_qubits
        )
    return devices


def get_quantum_kernel(feature_map, quantum_instance):
    """Get a quantum kernel

    Return an instance of FidelityQuantumKernel or
    FidelityStatevectorKernel (in the case of a simulation).

    Parameters
    ----------
    feature_map: QuantumCircuit | FeatureMap
        An instance of a feature map.
    quantum_instance: BaseSampler
        A instance of BaseSampler.

    Returns
    -------
    kernel: QuantumKernel
        The quantum kernel.

    Notes
    -----
    .. versionadded:: 0.3.0
    """
    if isinstance(quantum_instance._backend, AerSimulator):
        logging.log(logging.WARN,
                    """FidelityQuantumKernel skip for time constraint.
                    Using FidelityStatevectorKernel with AerStatevector.
                    Seed cannot be set with FidelityStatevectorKernel.
                    Increase the number of shots to diminish the noise."""
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
