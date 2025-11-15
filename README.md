[![Qiskit Ecosystem](https://img.shields.io/endpoint?style=flat&url=https%3A%2F%2Fqiskit.github.io%2Fecosystem%2Fb%2Ffb1907a7)](https://qisk.it/e)
[![PyPI version](https://badge.fury.io/py/pyriemann-qiskit.svg)](https://badge.fury.io/py/pyriemann-qiskit)
[![Documentation Status](https://readthedocs.org/projects/pyriemann-qiskit/badge/?version=latest)](https://pyriemann-qiskit.readthedocs.io/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)

# pyRiemann-qiskit

**Quantum-enhanced machine learning for Brain-Computer Interfaces and EEG analysis**

pyRiemann-qiskit bridges quantum computing (via Qiskit) with Riemannian geometry (via
pyRiemann) to enable quantum classification algorithms for BCI and EEG data. The library
implements quantum versions of Support Vector Classifiers (QSVC), Variational Quantum
Classifiers (VQC), and the Nearest Centroid Hypersphere (NCH) algorithm with convex
optimization.

## Table of Contents

- [Key Features](#key-features)
- [Overview](#overview)
- [Limitations & Considerations](#limitations--considerations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#how-to-cite)
- [References](#references)
- [Links & Resources](#links--resources)
- [Troubleshooting](#troubleshooting)

## Key Features

- 🔬 **Quantum Classifiers**: QSVC and VQC implementations with Riemannian geometry
- 🧠 **Quantum NCH**: Quantum Nearest Convex Hull (convex optimization with constraint
  programming)
- 📊 **BCI/EEG Focus**: Optimized for covariance matrix classification
- 🔗 **Scikit-learn Compatible**: Follows scikit-learn API conventions
- 🎨 **Visualization Tools**: Quantum art and manifold visualization
- 🔥 **Firebase Integration**: Remote storage for experiment results

## Overview

Literature on quantum computing suggests it may offer an advantage compared with classical
computing in terms of computational time and outcomes, such as for pattern recognition or
when using limited training sets [1, 2].

A ubiquitous library on quantum computing is Qiskit [3]. Qiskit is an IBM library
distributed under Apache 2.0 which provides both quantum algorithms and backends. A
backend can be either your local machine or a remote machine, which can emulate or be a
quantum machine. Qiskit's abstraction over the type of machine you want to use makes
designing quantum algorithms seamless.

Qiskit implements a quantum version of support vector-like classifiers, known as
quantum-enhanced support vector classifiers (QSVCs) and variational quantum classifiers
(VQCs) [4]. These classifiers likely offer an advantage over classical SVM in situations
where the classification task is complex. Task complexity is raised by the encoding of the
data into a quantum state, the number of available data, and the quality of the data. An
initial study is available in [5], and it can be downloaded from
[here](doc/Presentations/QuantumERPClassification.pdf). Although there is no study on this
topic at the time of writing, this could be an interesting research direction to
investigate BCI illiteracy.

`pyRiemann-qiskit` implements a wrapper around QSVC and VQC, to use quantum classification
with Riemannian geometry. A use case would be to use vectorized covariance matrices in the
tangent space as an input for these classifiers, enabling a possible sandbox for
researchers and engineers in the field.

`pyRiemann-qiskit` also introduces a quantum version of the NCH algorithm with convex
optimization. See a dedicated example on quantum-NCH
[here](https://github.com/pyRiemann/pyRiemann-qiskit/blob/main/examples/ERP/noplot_classify_P300_nch.py).

The remainder of this README details some of the quantum limitations and will guide you
through installation. Full documentation, including API description, is available at
<https://pyriemann-qiskit.readthedocs.io/>. The repository also includes a
[wiki](https://github.com/pyRiemann/pyRiemann-qiskit/wiki) where you can find additional
information.

## Limitations & Considerations

### Hardware Constraints

The number of qubits (and therefore the feature dimension) is limited to:

- **Local simulator**: ~36 qubits (depends on system memory)
- **Open plan quantum hardware**: 127 qubits
- **Exploratory hardware**: 156 qubits (IBM Heron QPU, restricted access)

### Performance Considerations

- **Time complexity**: Quantum algorithms are computationally expensive, especially when
  running locally. A higher number of trials or dimensions increases the time to
  completion.
- **Queue times**: Remote backends may experience delays depending on the number of
  concurrent users.
- **Recommended use**: Offline analysis only

⚠️ **Note**: The number of trials is limited in the examples we provide for demonstration
purposes. However, you should avoid such practices in your own analysis and use sufficient
iterations for reliable results.

## Installation

### Prerequisites

- Python 3.10 - 3.12
- [Anaconda](https://www.anaconda.com/) (recommended)

### Basic Installation

**From PyPI (stable release):**

```bash
pip install pyriemann-qiskit
```

**From source (development version):**

```bash
git clone https://github.com/pyRiemann/pyRiemann-qiskit.git
cd pyRiemann-qiskit
pip install .
```

Note that the steps above need to be re-executed in your local environment after any
changes inside your local copy of the `pyriemann_qiskit` folder, including pulling from
remote.

**Verify installation:**

```python
import pyriemann_qiskit
print(pyriemann_qiskit.__version__)
```

### Optional Dependencies

**GPU optimization (Linux only):**

```bash
pip install .[optim_linux]
```

Note: Qiskit only provides binaries for Linux. For other platforms, or if you want to
enable specific NVIDIA optimization for quantum computing, you need to build the binary
[yourself](https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md#building-with-gpu-support).

**Symbolic simulation:**

```bash
pip install .[optim]
```

This enables [qiskit-symb](https://github.com/SimoneGasperini/qiskit-symb) integration.

**Documentation and examples:**

```bash
pip install .[docs]
```

**Testing:**

```bash
pip install .[tests]
```

### Running Examples

To run a specific example on your local machine, first install dependencies for
documentation:

```bash
pip install .[docs]
```

Then you can run the python example of your choice:

```bash
python examples/ERP/plot_classify_P300_bi.py
```

### Docker Installation

**Using VS Code Dev Containers:**

1. Install VS Code with "Remote - Containers" extension
2. Open project in VS Code
3. Press `F1` → "Rebuild and Reopen in Container"
4. Wait for the container to build
5. Verify installation:

```python
import pyriemann_qiskit
```

**Using Docker CLI:**

```bash
# Build image
docker build -t pyrq .

# Run container
docker run --detach pyrq
```

**Pre-built images:**

Docker images are available at
[GitHub Container Registry](https://github.com/pyRiemann/pyRiemann-qiskit/pkgs/container/pyriemann-qiskit).
They are pushed to the registry on each release.

## Quick Start

Here's a sketch example using P300 EEG data from MOABB:

```python
from moabb.datasets import bi2012
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pyriemann_qiskit.pipelines import QuantumClassifierWithDefaultRiemannianPipeline

# Load P300 dataset
paradigm = P300(resample=128)
dataset = bi2012()
dataset.subject_list = dataset.subject_list[0:2]  # Use 2 subjects for demo

# Create pipelines
pipelines = {}

# Quantum pipeline with Riemannian geometry
pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=512,  # Use None for classical SVM
    nfilter=2,
    dim_red=PCA(n_components=5),
    params={"n_jobs": 1}
)

# Classical pipeline for comparison
labels_dict = {"Target": 1, "NonTarget": 0}
pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]],
                     estimator="lwf", xdawn_estimator="scm"),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto")
)

# Evaluate
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset])
results = evaluation.process(pipelines)
print(results.groupby("pipeline").mean("score")[["score", "time"]])
```

For a complete working example, see
[examples/ERP/plot_classify_P300_bi.py](examples/ERP/plot_classify_P300_bi.py).

For more examples, see the [examples](examples/) directory.

## Documentation

Full documentation is available at <https://pyriemann-qiskit.readthedocs.io/>, including:

- API reference
- Tutorials and examples
- Theory and background
- Advanced usage

## Contributing

We welcome contributions! 🎉

### Ways to Contribute

1. **Report Issues**:
   [Create an issue](https://github.com/pyRiemann/pyRiemann-qiskit/issues/new) for bugs or
   feature requests
2. **Submit Pull Requests**: Fork, develop, and submit PRs against `master`
3. **Improve Documentation**: Help us make docs clearer
4. **Share Examples**: Contribute new use cases

### Development Guidelines

- Use `git rebase` instead of merge commits
- Follow scikit-learn conventions
- Run `flake8` for code formatting
- Add tests for new features
- Document all public methods (NumPy docstring format)

**Code contribution** can be on core functionalities, documentation, or automation:

- The core functionalities are based on `Python`,
  [pyRiemann](https://github.com/pyRiemann/pyRiemann),
  [Qiskit ML](https://github.com/Qiskit/qiskit-machine-learning) and follow the best
  practice from [scikit-learn](https://scikit-learn.org/stable/index.html).

**Check code style:**

```bash
pip install flake8
flake8
```

**Run tests:**

```bash
pip install .[tests]
pytest
```

You can also specify a specific test file:

```bash
pytest tests/test_classification.py
```

**Automated workflows**: GitHub Actions run automatically when you push a commit. However,
the workflow for example execution is only triggered when you modify one of the examples
or the documentation as the execution takes a lot of time. You can enable
[Github Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)
in your fork to see the result of the CI pipeline. Results are also indicated at the end
of your pull request when raised. However note, that workflows in the pull request need
approval from the maintainers before being executed.

See [AGENTS.md](AGENTS.md) for detailed development guidelines.

## How to cite?

Anton Andreev, Grégoire Cattan, Sylvain Chevallier, and Quentin Barthélemy.
'pyRiemann-qiskit: A Sandbox for Quantum Classification Experiments with Riemannian
Geometry'. Research Ideas and Outcomes 9 (20 March 2023).
https://doi.org/10.3897/rio.9.e101006.

This library is part of the [Qiskit Ecosystem](https://qiskit.org/ecosystem)

## References

[1] **Blance, A., & Spannowsky, M.** (2021). Quantum machine learning for particle physics
using a variational quantum classifier. _Journal of High Energy Physics_, 2021(2), 212.
https://doi.org/10.1007/JHEP02(2021)212

[2] **Rebentrost, P., Mohseni, M., & Lloyd, S.** (2014). Quantum Support Vector Machine
for Big Data Classification. _Physical Review Letters_, 113(13), 130503.
https://doi.org/10.1103/PhysRevLett.113.130503

[3] **Abraham, H., et al.** (2019). Qiskit: An Open-source Framework for Quantum
Computing. Zenodo. https://doi.org/10.5281/zenodo.2562110

[4] **Havlíček, V., et al.** (2019). Supervised learning with quantum-enhanced feature
spaces. _Nature_, 567(7747), 209–212. https://doi.org/10.1038/s41586-019-0980-2

[5] **Cattan, G., & Andreev, A.** (2022). First steps to the classification of ERPs using
quantum computation. _NTB Berlin 2022 - International Forum on Neural Engineering & Brain
Technologies_, Berlin, Germany. https://hal.archives-ouvertes.fr/hal-03672246/

## Links & Resources

- 📖 [Documentation](https://pyriemann-qiskit.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/pyRiemann/pyRiemann-qiskit/issues)
- 📝 [Wiki](https://github.com/pyRiemann/pyRiemann-qiskit/wiki)
- 🐳
  [Docker Images](https://github.com/pyRiemann/pyRiemann-qiskit/pkgs/container/pyriemann-qiskit)
- 🌐 [Qiskit Ecosystem](https://qisk.it/e)

## Troubleshooting

See our
[dedicated troubleshooting page](https://github.com/pyRiemann/pyRiemann-qiskit/wiki/Troubleshooting)
on the wiki.
