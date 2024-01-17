# -*- coding: utf-8 -*-
"""

Lists information for 3 real Quantum computers each with 127 qubits: 'ibm_brisbane','ibm_kyoto', 'ibm_osaka'

Author: Anton Andreev

"""
#from qiskit import *
from qiskit.providers.ibmq import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends(simulator=False)

backend_names = ['ibm_brisbane','ibm_kyoto', 'ibm_osaka']

for backend_name in backend_names:
    backend = provider.get_backend(backend_name)
    status = backend.status()
    print("Name: ", status.backend_name, ", Is operational:", status.operational, ", Pending jobs: ", status.pending_jobs, ", Status message", status.status_msg)