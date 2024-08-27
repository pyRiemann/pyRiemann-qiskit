from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.primitives import BackendSampler
from qiskit_algorithms.optimizers import SLSQP
from docplex.mp.model import Model
from qiskit_algorithms.optimizers import COBYLA, ADAM, SLSQP, SPSA
from pyriemann_qiskit.utils.docplex import QAOACVOptimizer
import matplotlib.pyplot as plt
import math


def create_mixer_rotational_X_gates(angle):
    def mixer_X(num_qubits):
        qr = QuantumRegister(num_qubits)
        mixer = QuantumCircuit(qr)

        for i in range(num_qubits):
            mixer.rx(angle, qr[i])

    return mixer_X


def create_mixer_rotational_XY_gates(angle):
    def mixer_XY(num_qubits):
        qr = QuantumRegister(num_qubits)
        mixer = QuantumCircuit(qr)

        for i in range(num_qubits - 1):
            mixer.rx(angle, qr[i])
            mixer.rx(angle, qr[i + 1])
            mixer.ry(angle, qr[i])
            mixer.ry(angle, qr[i + 1])

    return mixer_XY


def create_mixer_rotational_XZ_gates(angle):
    def mixer_XZ(num_qubits):
        qr = QuantumRegister(num_qubits)
        mixer = QuantumCircuit(qr)

        for i in range(1, num_qubits - 1):
            mixer.rz(angle, qr[i - 1])
            mixer.rx(angle, qr[i])
            mixer.rx(angle + math.pi / 2, qr[i])
            mixer.rz(angle, qr[i + 1])

    return mixer_XZ


def run_qaoa_cv(n_reps, optimizer, create_mixer):
    # define docplex model
    mdl = Model("docplex model")
    x = mdl.continuous_var(-1, 0, "x")
    y = mdl.continuous_var(0, 2, "y")
    z = mdl.continuous_var(1.1, 2.2, "z")
    mdl.minimize((x - 0.83 + y + 2 * z) ** 2)

    n_var = mdl.number_of_variables

    # Define the BackendSampler (previously QuantumInstance)
    backend = AerSimulator(method="statevector", cuStateVec_enable=True)
    quantum_instance = BackendSampler(
        backend, options={"shots": 200, "seed_simulator": 42}
    )
    quantum_instance.transpile_options["seed_transpiler"] = 42

    qaoa_cv = QAOACVOptimizer(create_mixer, n_reps, n_var, quantum_instance, optimizer)
    solution = qaoa_cv.solve(qp)

    print(f"time = {qaoa_cv.run_time_}")

    # running QAOA circuit with optimal parameters
    print(f"solution = {solution}")
    print(f"min = {qaoa_cv.minimum_}")

    plt.plot(qaoa_cv.x_, qaoa_cv.y_)

    print(qaoa_cv.state_vector_)

    plt.show()

    return (qaoa_cv.run_time_, qaoa_cv.minimum_)


# List of optimizers
maxiter = 1000
optimizers = [
    COBYLA(maxiter=maxiter),
    ADAM(maxiter=maxiter, lr=0.1, tol=1e-8, noise_factor=1e-3, amsgrad=True),
    SLSQP(maxiter=maxiter),
    SPSA(maxiter=maxiter),
]

repetitions = range(1, 5)

ret = {}

for angle in range(4):
    angle = math.pi / 4 * angle
    mixers = [
        create_mixer_rotational_X_gates(angle),
        create_mixer_rotational_XY_gates(angle),
        create_mixer_rotational_XZ_gates(angle),
    ]
    for opt in optimizers:
        for rep in repetitions:
            for create_mixer in mixers:
                print(
                    f"Running QAOA with angle {angle}, optimizer {type(opt).__name__}, {rep} repetitions and {create_mixer.__name__} method"
                )
                key = f"{angle}_{type(opt).__name__}_{rep}_{create_mixer.__name__}"
                ret[key] = run_qaoa_cv(rep, opt, create_mixer)
