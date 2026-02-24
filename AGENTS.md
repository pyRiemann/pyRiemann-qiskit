# AI Agent Guidelines for pyRiemann-qiskit

This document provides guidance for AI coding agents working on the pyRiemann-qiskit
repository.

## Project Overview

**pyRiemann-qiskit** is a Python library that bridges Riemannian geometry (via pyRiemann)
with quantum computing (via Qiskit) for machine learning applications. The library focuses
on:

- Quantum-enhanced classification for Brain-Computer Interface (BCI) and EEG data
- Quantum Support Vector Classifiers (QSVC) and Variational Quantum Classifiers (VQC)
- Quantum implementations of the Minimum Distance to Mean (MDM) algorithm
- Integration with Riemannian geometry for covariance matrix processing

## Technology Stack

- **Core**: Python 3.10-3.12
- **Quantum Computing**: Qiskit 1.x, Qiskit Machine Learning 0.7.2, Qiskit Algorithms
  0.3.1
- **Riemannian Geometry**: pyRiemann 0.9
- **Machine Learning**: scikit-learn 1.5.2
- **Optimization**: CVXPY 1.6.5, DOcplex 2.29.245
- **Scientific Computing**: NumPy <2.3, SciPy 1.13.1
- **Testing**: pytest
- **Documentation**: Sphinx, sphinx-gallery
- **Code Quality**: flake8

## Repository Structure

```
pyriemann_qiskit/
├── classification/       # Quantum classification algorithms and wrappers
├── datasets/            # Dataset utilities
├── utils/               # Core utilities (distance, mean, preprocessing, etc.)
├── visualization/       # Visualization tools (manifold, quantum art)
├── autoencoders.py      # Quantum autoencoder implementations
├── ensemble.py          # Ensemble methods
└── pipelines.py         # ML pipeline utilities

examples/
├── ERP/                 # Event-Related Potential examples
├── MI/                  # Motor Imagery examples
├── other_datasets/      # Financial data, Titanic, etc.
├── resting_states/      # Resting state analysis
└── toys_dataset/        # Toy dataset examples

tests/                   # Unit tests
doc/                     # Sphinx documentation
benchmarks/              # Performance benchmarks
```

## Key Design Patterns

- **scikit-learn interface**: All classifiers implement `fit` / `predict` / `score`. New
  classifiers should subclass `QuanticClassifierBase` in `classification/wrappers.py`.
- **Optimizer hierarchy**: `pyQiskitOptimizer` (in `utils/docplex.py`) is the abstract
  base. Subclasses override `spdmat_var` (variable type), `_solve_qp` (solve logic), and
  `get_weights`. Concrete subclasses: `ClassicalOptimizer`, `NaiveQAOAOptimizer`,
  `QAOACVOptimizer`, `QAOACVAngleOptimizer`.
- **Quantum backend abstraction**: `utils/quantum_provider.py` → `get_simulator()` returns
  an AerSimulator by default. Callers should not import backends directly.
- **Docplex ↔ Qiskit bridge**: Optimization problems are defined as Docplex models,
  converted via `from_docplex_mp`, then solved with QAOA or classical Cobyla.
- **Hyperparameter factory**: `utils/hyper_params_factory.py` centralises circuit
  configuration (feature maps, ansätze, mixers). Use `create_mixer_rotational_X_gates` for
  QAOA mixers.
- **Symbolic statevector cache**: Pre-computed statevectors live in `symb_statevectors/`
  (XFeatureMap, ZFeatureMap, ZZFeatureMap). Do not delete or regenerate these unless
  intentional.

## Key Constraints & Considerations

### Quantum Computing Limitations

1. **Qubit Limitations**:

   - Local simulator: ~36 qubits (memory-dependent)
   - Remote simulator: up to 5000 qubits
   - Free quantum hardware: 7 qubits
   - Exploratory hardware: 127 qubits (not public)

2. **Time Complexity**: Quantum algorithms are computationally expensive, especially
   locally. Use limited trials in examples but note this limitation.

3. **Offline Only**: Due to time constraints and queuing on remote backends, this library
   is designed for offline analysis only.

### Python Version Support

- **Supported**: Python 3.10 - 3.12
- Always ensure compatibility across this range

### Platform Support

- Windows, macOS, and Linux
- GPU optimization only available on Linux (qiskit-aer-gpu)
- Symbolic simulation available via qiskit-symb

## Development Guidelines

### Code Style

1. **Formatting**: Use `flake8` for code formatting

   ```bash
   flake8
   ```

2. **Follow scikit-learn conventions**: The library follows scikit-learn best practices
   for estimators and transformers

3. **Documentation**: Always provide full documentation for public methods using NumPy
   docstring format

### Testing

1. **Run tests locally**:

   ```bash
   pip install .[tests]
   pytest
   ```

2. **Test specific files**:

   ```bash
   pytest tests/test_classification.py
   ```

3. **Coverage**: Maintain test coverage for new features

4. **Test fixtures** (from `tests/conftest.py`):
   - `get_covmats(n_matrices, n_channels)` — generates SPD covariance matrices
   - `get_labels(n_matrices, n_classes)` — generates classification labels
   - `get_dataset(kind)` — generates full `(X, y)` datasets; kinds: `rand`, `bin`,
     `rand_cov`, `bin_cov`
   - `requires_matplotlib` / `requires_seaborn` — decorators for optional-dependency tests
   - Use `BinaryTest` / `MultiClassTest` base classes for classifier tests

### Git Workflow

1. **Avoid merge commits**: Use `git rebase` instead of merging master into feature
   branches
2. **Pull Requests**: Always create PRs against the master branch
3. **CI/CD**: GitHub Actions run automatically on push (examples only run when modified
   due to execution time)

### Documentation

1. **Sphinx-based**: Documentation uses Sphinx with sphinx-gallery for examples
2. **API Documentation**: Auto-generated from docstrings
3. **Examples**: Place in appropriate subdirectory under `examples/`
4. **Build docs**:
   ```bash
   pip install .[docs]
   cd doc
   make html
   ```

## Common Tasks

### Adding a New Classifier

1. Implement in `pyriemann_qiskit/classification/algorithms.py` or `wrappers.py`
2. Follow scikit-learn estimator interface (`fit`, `predict`, `score`)
3. Add comprehensive docstrings
4. Create unit tests in `tests/test_classification.py`
5. Add example in appropriate `examples/` subdirectory
6. Update API documentation in `doc/api.rst`

### Adding a New Utility Function

1. Place in appropriate module under `pyriemann_qiskit/utils/`
2. Export in `pyriemann_qiskit/utils/__init__.py`
3. Add tests in corresponding test file
4. Document thoroughly

### Adding Examples

1. Place in appropriate subdirectory under `examples/`
2. Follow sphinx-gallery format (docstring at top with RST formatting)
3. Use limited trials for quantum algorithms (note this in comments)
4. Include README.txt in the subdirectory
5. Examples should be runnable with: `pip install .[docs]`
6. Prefix filename with `noplot_` to exclude an example from sphinx-gallery rendering
   while still running it in CI (use this for long-running experiments)

### Working with Dependencies

1. **Core dependencies**: Listed in `setup.py` `install_requires`
2. **Optional dependencies**: Use `extras_require` for docs, tests, optimization
3. **Version pinning**: Critical dependencies are pinned (e.g., qiskit versions)
4. **NumPy**: Keep <2.3 for compatibility

## Important Files

- `setup.py`: Package configuration and dependencies
- `setup.cfg`: Additional package metadata
- `pyriemann_qiskit/_version.py`: Version string
- `requirements.txt`: Development requirements
- `.pre-commit-config.yaml`: Pre-commit hooks
- `ecosystem.json`: Qiskit Ecosystem metadata
- `symb_statevectors/`: Cached symbolic statevectors for XFeatureMap, ZFeatureMap,
  ZZFeatureMap — do not delete

## Integration Points

### Firebase Integration

- `pyriemann_qiskit/utils/firebase_connector.py`: Firebase integration for remote storage
- `pyriemann_qiskit/utils/firebase_cert.py`: Certificate handling
- Used for storing/retrieving experiment results

### Quantum Providers

- `pyriemann_qiskit/utils/quantum_provider.py`: Abstraction for quantum backends
- Supports local simulators, remote simulators, and real quantum hardware

### Hyperparameter Management

- `pyriemann_qiskit/utils/hyper_params_factory.py`: Factory for quantum algorithm
  hyperparameters
- Centralizes configuration for different quantum circuits and algorithms

## Testing Strategy

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test pipelines and workflows
3. **Mock Quantum Backends**: Use simulators for testing, not real hardware
4. **Fast Tests**: Keep test execution time reasonable

## Documentation Standards

### General Documentation Principles

1. **Skip `__init__.py` files**: Do not add or modify docstrings in `__init__.py` files
2. **Skip private methods**: Do not document private methods (those starting with `_`) or
   constructors (`__init__`)
3. **Document public methods only**: Focus on public methods and class docstrings
4. **Keep original style**: Follow the original spirit of the writing - be direct and
   technical
5. **Avoid marketing language**: Do not use words like "enhanced", "improved", "powerful",
   etc.
6. **Be concise and technical**: Get straight to the point without unnecessary elaboration

### Docstring Format (NumPy Style)

```python
def function_name(param1, param2):
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    References
    ----------
    .. [1] Citation if applicable
    """
```

### What to Document

**DO document:**

- Module-level docstrings (at the top of `.py` files, not `__init__.py`)
- Public class docstrings
- Public method docstrings (`fit`, `predict`, `transform`, `score`, etc.)
- Public function docstrings
- Property docstrings

**DO NOT document:**

- `__init__.py` files
- Private methods (starting with `_`)
- Constructor methods (`__init__`)
- Internal helper functions

### Writing Style Guidelines

**Good examples:**

```python
"""Quantum classifier wrappers.

Contains the base class for all quantum classifiers and several quantum
classifiers that can run in quantum/classical modes and on simulated/real
quantum computers.
"""
```

```python
def predict(self, X):
    """Predict class labels for samples in X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Test samples.

    Returns
    -------
    y_pred : ndarray, shape (n_samples,)
        Predicted class labels.
    """
```

**Bad examples (avoid):**

```python
"""Enhanced quantum classifier wrappers with improved performance.

This module provides powerful quantum classifiers that leverage
state-of-the-art quantum computing to deliver superior results...
"""
```

```python
def predict(self, X):
    """Calculates the predictions using our advanced algorithm.

    This method uses cutting-edge quantum technology to provide
    highly accurate predictions...
    """
```

### Example Format (Sphinx-Gallery)

```python
"""
Title of Example
================

Description of what the example demonstrates.

.. note::
    Important notes about the example.
"""

# %%
# Section Header
# --------------
# Description of this section

# Code here
```

## References & Resources

- **Documentation**: https://pyriemann-qiskit.readthedocs.io/
- **Repository**: https://github.com/pyRiemann/pyRiemann-qiskit
- **Issues**: https://github.com/pyRiemann/pyRiemann-qiskit/issues
- **Wiki**: https://github.com/pyRiemann/pyRiemann-qiskit/wiki
- **Qiskit Ecosystem**: https://qisk.it/e

## Citation

When referencing this library:

> Anton Andreev, Grégoire Cattan, Sylvain Chevallier, and Quentin Barthélemy.
> 'pyRiemann-qiskit: A Sandbox for Quantum Classification Experiments with Riemannian
> Geometry'. Research Ideas and Outcomes 9 (20 March 2023).
> https://doi.org/10.3897/rio.9.e101006.

## Contact & Contribution

- **Maintainer**: @gcattan
- **License**: BSD 3-Clause
- **Contributions**: Welcome via issues and pull requests

## Agent-Specific Notes

1. **Quantum Simulation**: Be aware that quantum simulations can be slow. When creating
   examples, use small datasets and limited iterations.

2. **Version Compatibility**: Always check compatibility with pinned versions, especially
   Qiskit components.

3. **Platform Differences**: GPU optimization is Linux-only. Don't assume GPU
   availability.

4. **Documentation First**: This is a research-oriented library. Good documentation is
   critical.

5. **Test Coverage**: Maintain high test coverage, but use mocked quantum backends to keep
   tests fast.

6. **Breaking Changes**: Be cautious with changes to public APIs. This library is used in
   research.

7. **Performance**: Quantum algorithms are inherently slow. Focus on correctness over
   optimization unless specifically addressing performance.

8. **Examples as Documentation**: Examples serve as both tutorials and integration tests.
   Keep them clear and well-commented.
