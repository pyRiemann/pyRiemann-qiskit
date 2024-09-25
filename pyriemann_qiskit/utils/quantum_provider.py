"""Module containing helpers for IBM quantum backends
   providers and simulators."""

import logging
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.quantum_info import AerStatevector
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.kernels import (
    FidelityStatevectorKernel,
    FidelityQuantumKernel,
)
from qiskit.circuit import Parameter, ParameterVector
from qiskit_symb.quantum_info import Statevector


import numpy as np
import joblib
import asyncio
import pickle

class SymbFidelityStatevectorKernel:

    def __init__(self, circuit):
        self.circuit = circuit
        print(self.circuit)
        if isinstance(self.circuit, str):
            file = open(self.circuit,'rb')
            sv = pickle.load(file)
        else:
            sv = Statevector(self.circuit)
        print("Statevector created")
        # file = open("binary.dat",'wb')
        # pickle.dump(sv, file)
        self.function = sv.to_lambda()
        print("lambdify ok")
        self.n = 0

    def evaluate(self, x_vec, y_vec=None):
        if y_vec is None:
            y_vec = x_vec

        x_vec_len = len(x_vec)
        y_vec_len = len(y_vec)

        

        is_sim = x_vec_len == y_vec_len and (x_vec == y_vec).all()
        
        print(x_vec_len, y_vec_len, is_sim, self.n)
        self.n = self.n+1
        
        kernel_matrix = np.zeros((x_vec_len, y_vec_len))
        
        n_thread = 1
        chunck = x_vec_len // n_thread

        def compute_fidelity_partial_matrix(i_thread):
            
            # for i, x in enumerate(x_vec[i_thread * chunck: (i_thread + 1) * chunck]):
            for i in range(i_thread * chunck, (i_thread + 1) * chunck):
                x = x_vec[i]
                for j in range(i if is_sim else y_vec_len):
                    y = y_vec[j]
                # for j, y in enumerate(y_vec[:i+1] if is_sim else y_vec):
                    fidelity = abs(self.function(*x, *y)[0, 0]) ** 2
                    kernel_matrix[i, j] = fidelity
                    if is_sim:
                        kernel_matrix[j, i] = fidelity
            return kernel_matrix
        
        return compute_fidelity_partial_matrix(0)
        results = joblib.Parallel( n_jobs = n_thread )( joblib.delayed( compute_fidelity_partial_matrix )( i_thread )
                                                           for i_thread in range(n_thread)
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


def get_quantum_kernel(feature_map, gen_feature_map, quantum_instance, use_fidelity_state_vector_kernel):
    """Get a quantum kernel

    Return an instance of FidelityQuantumKernel or
    FidelityStatevectorKernel (in the case of a simulation).

    For simulation with a small number of qubits (< 9), qiskit-symb is used.

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
    .. versionchanged:: 0.4.0
        Add support for qiskit-symb
    """
    if use_fidelity_state_vector_kernel and isinstance(
        quantum_instance._backend, AerSimulator
    ):

        # For simulation:
        print(feature_map.num_qubits)
        if feature_map.num_qubits <= 9:
            # With a small number of qubits, let's use qiskit-symb
            # See:
            # https://medium.com/qiskit/qiskit-symb-a-qiskit-ecosystem-package-for-symbolic-quantum-computation-b6b4407fa705
            circuit = feature_map.compose(feature_map.inverse()).decompose()
            from qiskit.circuit.library import PauliFeatureMap

            # fm1 = PauliFeatureMap(
            #     feature_dimension=feature_map.num_qubits,
            #     paulis=["X"],
            #     data_map_func=None,
            #     parameter_prefix="a",
            #     insert_barriers=False,
            #     name="xfm",
            # )

            # fm2 = PauliFeatureMap(
            #     feature_dimension=feature_map.num_qubits,
            #     paulis=["X"],
            #     data_map_func=None,
            #     parameter_prefix="b",
            #     insert_barriers=False,
            #     name="xfm",
            # )
            # original_parameters = feature_map.ordered_parameters
            # num_param = len(original_parameters)
            # parameters_2 = ParameterVector("b")
            # feature_map.ordered_parameters = parameters_2
            # fm2 = feature_map.inverse()
            # feature_map.ordered_parameters =ParameterVector("a")
            fm2 = gen_feature_map(feature_map.num_qubits, "b")
            circuit = feature_map.compose(fm2.inverse()).decompose()
            print(circuit.num_qubits)
            key = f"{feature_map.name}-{feature_map.reps}"
            import os
            # circuit = key if os.path.isfile(key) else circuit
            kernel = SymbFidelityStatevectorKernel(circuit=circuit)
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
