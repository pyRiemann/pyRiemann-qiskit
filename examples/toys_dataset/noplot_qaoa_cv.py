from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.primitives import BackendSampler

from qiskit_algorithms.optimizers import SLSQP
from qiskit_optimization.problems import VarType
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

from qiskit.circuit.library import QAOAAnsatz

from qiskit_algorithms.optimizers import COBYLA, ADAM, SLSQP, SPSA

import matplotlib.pyplot as plt
import time
import math

from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
import math


def prepare_model(qp):
  scalers = []
  for v in qp.variables:
    if(v.vartype == VarType.CONTINUOUS):
      scaler = MinMaxScaler().fit(np.array([v.lowerbound, v.upperbound]).reshape(-1, 1))
      # print(scaler.data_min_, scaler.data_max_)
      scalers.append(scaler)
      v.vartype = VarType.BINARY
  return scalers


def create_mixer_rotational_X_gates(angle):
    def mixer_X(num_qubits):
      qr = QuantumRegister(num_qubits)
      mixer = QuantumCircuit(qr)

      for i in range(num_qubits):
        mixer.rx(angle,qr[i])

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
        mixer.rx(angle + math.pi/2, qr[i])
        mixer.rz(angle, qr[i + 1])

    return mixer_XZ


def run_qaoa_cv(n_reps, optimizer, create_mixer):

  # define docplex model
  mdl = Model("docplex model")
  x = mdl.continuous_var(-1, 0, "x")
  # x = mdl.continuous_var(0, 1, "x")
  y = mdl.continuous_var(0, 2, "y")
  z = mdl.continuous_var(1.1, 2.2, "z")
  mdl.minimize((x - 0.83 + y + 2 * z)**2)
  # mdl.minimize((x - 0.83) ** 2)

  n_var = mdl.number_of_variables

  # convert docplex model to quadratic program
  qp = from_docplex_mp(mdl)

  # Extract the objective function from the docplex model
  # We want the object expression with continuous variable
  objective_expr = qp._objective

  # Convert continous variable to binary ones
  # Get scalers corresponding to the definition range of each variables
  scalers = prepare_model(qp)

  # Check all variables are converted to binary, and scalers are registered
  # print(qp.prettyprint(), scalers)

  # cost operator
  # Get operator associated with model
  cost, offset = qp.to_ising()

  mixer = create_mixer(cost.num_qubits)

  # QAOA cirtcuit
  ansatz_0 = QAOAAnsatz(cost_operator=cost, reps=n_reps, initial_state=None, mixer_operator=mixer).decompose()
  ansatz = QAOAAnsatz(cost_operator=cost, reps=n_reps, initial_state=None, mixer_operator=mixer).decompose()
  ansatz.measure_all()

  # Define the BackendSampler (previously QuantumInstance)
  backend = AerSimulator(method="statevector", cuStateVec_enable=True)
  quantum_instance = BackendSampler(
                  backend, options={"shots": 200, "seed_simulator": 42}
              )
  quantum_instance.transpile_options["seed_transpiler"] = 42

  def prob(job, i):
      quasi_dists = job.result().quasi_dists[0]
      p = 0
      for key in quasi_dists:
        if(key & 2**(n_var - 1 - i)):
          p += quasi_dists[key]

      # p is in the range [0, 1].
      # We now need to scale it in the definition range of our continuous variables
      p = scalers[i].inverse_transform([[p]])[0][0]
      return p

  # defining loss function
  x = []
  y = []
  def loss(params):
      job = quantum_instance.run(ansatz, params)
      var_hat = [prob(job, i) for i in range(n_var)]
      cost = objective_expr.evaluate(var_hat)
      x.append(len(x))
      y.append(cost)
      return cost

  # Initial guess for the parameters
  initial_guess = np.array([1, 1] * n_reps)

  # minimize function to search for the optimal parameters
  # result = minimize(loss, initial_guess, method='COBYLA', options={"maxiter":1000})
  start_time = time.time()
  result = optimizer.minimize(loss, initial_guess)
  stop_time = time.time()
  run_time = stop_time - start_time
  print(f"time = {run_time}")
  optim_params = result.x

  # running QAOA circuit with optimal parameters
  job = quantum_instance.run(ansatz, optim_params)
  solution = [prob(job, i) for i in range(n_var)]
  minimum = objective_expr.evaluate(solution)
  print(f"solution = {solution}")
  print(f"min = {minimum}")

  plt.plot(x, y)

  optimized_circuit = ansatz_0.assign_parameters(optim_params)
  state = Statevector(optimized_circuit)
  print(state)

  plt.show()

  return (run_time, minimum)

# List of optimizers
maxiter=1000
optimizers = [
    COBYLA(maxiter=maxiter),
    ADAM(maxiter=maxiter, lr=0.1, tol=1e-8, noise_factor=1e-3, amsgrad=True),
    SLSQP(maxiter=maxiter),
    SPSA(maxiter=maxiter)
    ]

# no. of repetitions
repetitions = range(1, 5)

ret = {}


for angle in range(4):
  angle = math.pi / 4 * angle
  mixers = [create_mixer_rotational_X_gates(angle), create_mixer_rotational_XY_gates(angle), create_mixer_rotational_XZ_gates(angle)]
  for opt in optimizers:
      for rep in repetitions:
          for create_mixer in mixers:
            print(f"Running QAOA with angle {angle}, optimizer {type(opt).__name__}, {rep} repetitions and {create_mixer.__name__} method")
            key = f"{angle}_{type(opt).__name__}_{rep}_{create_mixer.__name__}"
            ret[key] = run_qaoa_cv(rep, opt, create_mixer)

