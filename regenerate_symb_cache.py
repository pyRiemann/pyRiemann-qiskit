"""Regenerate all cached symbolic statevectors in symb_statevectors/.

Edit CONFIGS below to match the (generator, reps, n_qubits) combinations
you need, then run from the project root:
    python regenerate_symb_cache.py
"""

import os

from pyriemann_qiskit.utils.hyper_params_factory import (
    gen_x_feature_map,
    gen_z_feature_map,
    gen_zz_feature_map,
)
from pyriemann_qiskit.utils.quantum_provider import SymbFidelityStatevectorKernel

# Each entry: (generator_factory, reps, n_qubits)
CONFIGS = [
    (gen_func, reps, n_qubits)
    for gen_func in [gen_x_feature_map, gen_z_feature_map, gen_zz_feature_map]
    for reps in [2, 3]
    for n_qubits in [2, 3, 4]
]

DIR = "symb_statevectors"


def regenerate_all():
    # Delete all existing cache files first (may be incompatible with current Qiskit)
    if os.path.isdir(DIR):
        for filename in os.listdir(DIR):
            path = os.path.join(DIR, filename)
            print(f"Removing stale cache: {filename}")
            os.remove(path)

    for gen_factory, reps, n_qubits in CONFIGS:
        gen_func = gen_factory(reps=reps)
        feature_map = gen_func(n_qubits)
        print(f"Generating {feature_map.name}-{reps} (n_qubits={n_qubits}) ...")
        SymbFidelityStatevectorKernel(feature_map, gen_func)

    print("Done.")


if __name__ == "__main__":
    regenerate_all()
