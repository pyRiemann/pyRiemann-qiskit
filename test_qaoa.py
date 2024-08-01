  !pip install qiskit
  !pip install qiskit-aer
  !pip install qiskit-optimization
  !pip install docplex
  import qiskit_optimization

from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.primitives import BackendSampler

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms import QAOA
from qiskit_aer import Aer
from qiskit_optimization.problems import QuadraticProgram, VarType
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize

from qiskit_algorithms.optimizers import COBYLA, ADAM, SLSQP, SPSA

import matplotlib.pyplot as plt
import time
import math

from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city
from sklearn.preprocessing import MinMaxScaler
from qiskit.quantum_info import SparsePauliOp

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
import math
from scipy import (zeros, array, arange, exp, real, conj, pi,
                   copy, sqrt, meshgrid, size, polyval, fliplr, conjugate,
                   cos, sin)
from scipy.integrate import simps

  # Define the problem instance
  mdl = Model("docplex model")
  x = mdl.binary_var("x")
  mdl.minimize((x - 0.83)**2)

  qp = from_docplex_mp(mdl)
  print(qp.prettyprint())

  # Define the optimizer
  optimizer = SLSQP()  # Replace with your optimizer

  # Define the BackendSampler (previously QuantumInstance)
  backend = AerSimulator(method="statevector", cuStateVec_enable=True)
  quantum_instance = BackendSampler(
                  backend, options={"shots": 200, "seed_simulator": 42}
              )
  quantum_instance.transpile_options["seed_transpiler"] = 42


  # Define the QAOA measurement setting
  qaoa_mes = QAOA(
      optimizer=optimizer,
      sampler=quantum_instance)

  # Create a MinimumEigenOptimizer object
  qaoa = MinimumEigenOptimizer(qaoa_mes)

  # Solve the optimization problem
  qaoa_result = qaoa.solve(qp)

  qaoa_ansatz = qaoa_mes.ansatz
  print(qaoa_ansatz)

  # Print the result
  print(qaoa_result)


print(qaoa_result.min_eigen_solver_result.optimal_point)
print(qaoa_result.min_eigen_solver_result.optimal_parameters)

print(qaoa_result.samples)
quantum_instance.run(qaoa_ansatz, qaoa_result.min_eigen_solver_result.optimal_point)

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
  # plot_state_city(state, alpha=0.6)
  # plot_wigner_state_circuit(optimized_circuit)

  plt.show()

  return (run_time, minimum)

# run_qaoa_cv(2, COBYLA(maxiter=500), None)

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

print(ret)

ret

def wigner(rho, xvec, yvec, g=math.sqrt(2)):
    """
    https://qutip.org/docs/4.0.2/modules/qutip/wigner.html#wigner
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.

    The Wigner function is calculated as
    :math:`W = \sum_{mn} \\rho_{mn} W_{mn}` where :math:`W_{mn}` is the Wigner
    function for the density matrix :math:`|m><n|`.

    In this implementation, for each row m, Wlist contains the Wigner functions
    Wlist = [0, ..., W_mm, ..., W_mN]. As soon as one W_mn Wigner function is
    calculated, the corresponding contribution is added to the total Wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """

    M = np.prod(rho.shape[0])
    X, Y = meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)

    Wlist = array([zeros(np.shape(A), dtype=complex) for k in range(M)])
    Wlist[0] = exp(-2.0 * abs(A) ** 2) / pi

    W = real(rho[0, 0]) * real(Wlist[0])
    for n in range(1, M):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / sqrt(n)
        W += 2 * real(rho[0, n] * Wlist[n])

    for m in range(1, M):
        temp = copy(Wlist[m])
        Wlist[m] = (2 * conj(A) * temp - sqrt(m) * Wlist[m - 1]) / sqrt(m)

        # Wlist[m] = Wigner function for |m><m|
        W += real(rho[m, m] * Wlist[m])

        for n in range(m + 1, M):
            temp2 = (2 * A * Wlist[n - 1] - sqrt(m) * temp) / sqrt(n)
            temp = copy(Wlist[n])
            Wlist[n] = temp2

            # Wlist[n] = Wigner function for |m><n|
            W += 2 * real(rho[m, n] * Wlist[n])

    return 0.5 * W * g ** 2

def plot_wigner_state_circuit(qc):

  rho = DensityMatrix(qc)
  # rho = Statevector(qc)
  print(rho.data.shape[0])

  # rho_a = partial_trace(state=rho, qargs=[1, 2])

  a = 300
  final = wigner(rho.data, [i/100 for i in range(-a, a)], [i/100 for i in range(-a,a)])

  x = np.array([i/10 for i in range(-a, a)])
  y = []


  for i in range(0, len(x)):
      res = simps([final[k][i] for k in range(0, len(x))], x)
      y.append(res)

  y = np.array(y)

  opt = x[y == y.max()][0]
  print((opt + a/10) / (2*a/10))
  plt.plot(x, y)

run_qaoa_cv(4, COBYLA(), create_mixer_X_gates)
