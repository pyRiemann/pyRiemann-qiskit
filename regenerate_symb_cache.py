"""Regenerate all cached symbolic statevectors in symb_statevectors/.

Cache filenames are {feature_map.name}-{reps}-{n_qubits} (see
SymbFidelityStatevectorKernel.__init__). Cached pickles embed Qiskit
Rust-backed objects (e.g. ParameterVector/ParameterVectorElement), whose
pickle format is an internal implementation detail that can change between
Qiskit versions -- this is why cached files can suddenly fail to unpickle
after a Qiskit bump ("state is not a dictionary").

Rather than hardcoding the (generator, reps, n_qubits) combinations, this
script parses the existing cache filenames to know exactly what to
regenerate, so no cached configuration is silently dropped. To add a new
combination, add an extra entry to EXTRA_CONFIGS below, then run from the
project root:
    python regenerate_symb_cache.py
"""

import os

from pyriemann_qiskit.utils.hyper_params_factory import (
    gen_x_feature_map,
    gen_z_feature_map,
    gen_zz_feature_map,
)
from pyriemann_qiskit.utils.quantum_provider import SymbFidelityStatevectorKernel

DIR = "symb_statevectors"

GEN_FACTORIES_BY_NAME = {
    "XFeatureMap": gen_x_feature_map,
    "ZFeatureMap": gen_z_feature_map,
    "ZZFeatureMap": gen_zz_feature_map,
}

# Extra (generator_factory, reps, n_qubits) combinations to (re)generate
# in addition to whatever is already cached on disk.
EXTRA_CONFIGS = []


def _configs_from_existing_cache():
    if not os.path.isdir(DIR):
        return []

    configs = []
    for filename in os.listdir(DIR):
        name, reps, n_qubits = filename.rsplit("-", 2)
        gen_factory = GEN_FACTORIES_BY_NAME[name]
        configs.append((gen_factory, int(reps), int(n_qubits)))
    return configs


def regenerate_all():
    configs = sorted(
        set(_configs_from_existing_cache() + EXTRA_CONFIGS),
        key=lambda cfg: (cfg[0].__name__, cfg[1], cfg[2]),
    )

    # Delete all existing cache files first (may be incompatible with current Qiskit)
    if os.path.isdir(DIR):
        for filename in os.listdir(DIR):
            path = os.path.join(DIR, filename)
            print(f"Removing stale cache: {filename}")
            os.remove(path)

    for gen_factory, reps, n_qubits in configs:
        gen_func = gen_factory(reps=reps)
        feature_map = gen_func(n_qubits)
        print(f"Generating {feature_map.name}-{reps} (n_qubits={n_qubits}) ...")
        SymbFidelityStatevectorKernel(feature_map, gen_func)

    print("Done.")


if __name__ == "__main__":
    regenerate_all()
