# pyRiemann-qiskit

Litterature on quantum computing suggests it may offer an advantage as compared
with classical computing in terms of computational time and outcomes, such as
for pattern recognition or when using limited training sets [1, 2].

A ubiquitous library on quantum computing is Qiskit [3].
Qiskit is an IBM library distributed under Apache 2.0 which provides both
quantum algorithms and backends. A backend can be either your local machine
or a remote machine, which one can emulates or be a quantum machine.
Qiskit abstraction over the type of machine you want to use, make designing
quantum algorithm seamless.

Qiskit implements a quantum version of support vector
-like classifiers, known as quantum-enhanced support vector classifier (QSVC)
and varitional quantum classifier (VQC) [4]. These classifiers likely offer
an advantage over classical SVM in situations where the classification task
is complex. Task complexity is raised by the encoding of the data into a
quantum state, the number of available data and the quality of the data. An initial 
study is available in [5], and it can be downloaded from [here](doc/Presentations/QuantumERPClassification.pdf).
Although there is no study on this topic at the time of writting,
this could be an interesting research direction to investigate BCI illiteracy.

pyRiemann-qiskit implements a wrapper around QSVC and VQC, to use quantum
classification with Riemannian geometry. A use case would be to use vectorized
covariance matrices in the tangent space as an input for these classifiers,
enabling a possible sandbox for researchers and engineers in the field.

The remaining details some of the quantum drawbacks and will guide you through installation.
Full documentation, including API description, is available at <https://pyriemann-qiskit.readthedocs.io/>.
The repository also includes a [wiki](https://github.com/pyRiemann/pyRiemann-qiskit/wiki) where you can find additional information.

## Quantum drawbacks

- Limitation of the feature dimension

    The number of qubits (and therefore the feature dimension) is limited to:
    - 24 on a local quantum simulator, and up to:
    - 5000 on a remote quantum simulator;
    - 5 on free real quantum computers, and up to:
    - 65 on exploratory quantum computers (not available for public use).

- Time complexity

    A higher number of trials or dimension increases time to completion of the quantum algorithm, especially when running on a local machine. This is why the number of trials is limited in the examples we provided. However, you should avoid such practices in your own analysis. 
    
    Although these aspects are less important in a remote backend, it may happen that the quantum algorithm is queued depending on the number of concurrent users.

    For all these aspects, the use of pyRiemann-qiskit should be limited to offline analysis only.
    
## References

[1] A. Blance and M. Spannowsky,
    ‘Quantum machine learning for particle physics using a variational quantum classifier’,
    J. High Energ. Phys., vol. 2021, no. 2, p. 212, Feb. 2021,
    doi: 10.1007/JHEP02(2021)212.

[2] P. Rebentrost, M. Mohseni, and S. Lloyd,
   ‘Quantum Support Vector Machine for Big Data Classification’,
    Phys. Rev. Lett., vol. 113, no. 13, p. 130503, Sep. 2014,
    doi: 10.1103/PhysRevLett.113.130503.

[3] H. Abraham et al., Qiskit: An Open-source Framework for Quantum Computing.
    Zenodo, 2019. doi: 10.5281/zenodo.2562110.

[4] V. Havlíček et al.,
    ‘Supervised learning with quantum-enhanced feature spaces’,
    Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
    doi: 10.1038/s41586-019-0980-2.
	
[5] G. Cattan, A. Andreev,
    First steps to the classification of ERPs using quantum computation,
	NTB Berlin 2022 - International Forum on Neural Engineering & Brain Technologies, May 2022, Berlin, Germany,
	hal: https://hal.archives-ouvertes.fr/hal-03672246/


## Installation

_We recommend the use of [Anaconda](https://www.anaconda.com/) to manage python environements._ 

`pyRiemann-qiskit` currently supports Windows, Mac and Linux OS with Python 3.7, 3.8 and 3.9.

You can install `pyRiemann-qiskit` release from PyPI:

```
pip install pyriemann-qiskit
```

The development version can be installed by cloning this repository
and installing the package on your local machine using the `setup.py` script:

```
python setup.py develop
```

To check the installation, open a python shell and type:

```
import pyriemann_qiskit
```

To run a specific example on your local machine, you should install first dependencies for the documentation:

```
pip install .[docs]
```

Then you can run the python example of your choice like:

```
python examples\ERP\classify_P300_bi.py
```

### Installation with docker

We also offer the possibility to set up the dev environment within docker.
To this end, we recommand to use `vscode` with the `Remote Containers` extension
from Microsoft. 

Once the installation is successful, just open the project in `vscode` and enter `F1`.
In the search bar that opens, type `Rebuild and Reopen Container`.

Wait for the container to build, and open a python shell within the container.
Then ensure everything went smoothly by typing:

```
import pyriemann_qiskit
```

## Contributor Guidelines

Everyone is welcomed to contribute to this repository. There are two types of contributions:

- [Raise an issue](https://github.com/pyRiemann/pyRiemann-qiskit/issues/new) on the repository.
Issues can be either a bug report or an enhancement proposal. Note that it is necessary to register on
GitHub before. There is no special template which is expected but, if you raise a defect please  provide as much details as possible.

- [Raise a pull request](https://github.com/pyRiemann/pyRiemann-qiskit/compare). Fork the repository and work on your own branch. Then raise a pull request with your branch against master. As much as possible, we ask you to:
    - avoid merging master into your branch. Always prefer git rebase.
    - always provide full documentation of public method.

Code contribution (pull request) can be either on core functionalities, documentation or automation.

- The core functionalies are based on `Python`, [pyRiemann](https://github.com/pyRiemann/pyRiemann), [Qiskit ML](https://github.com/Qiskit/qiskit-machine-learning) and follow the best practice from [scikit-learn](https://scikit-learn.org/stable/index.html). We use `flake8` for code formatting. `flake8` is installed with the testing dependencies (see below) or can be installed directly from `pip`:

    ```
    pip install flake8
    ```

    To execute `flake8`, just type `flake8` from the root repository, and correct all errors related to your changes.

- The documentation is based on [Sphinx](https://www.sphinx-doc.org/en/master/).
- Automation is based on `GitHub Action` and `pytest`. It consists in two automated workflows for running the example and the tests. To run the tests on your local machine, you should first install the dependencies for testing:

    ```
    pip install .[tests]
    ```

    and then type `pytest` from the root repository. You can also specify a file like:

    ```
    pytest tests/test_classification.py 
    ```

    Workflows are automatically triggered when you push a commit. However, the worflow for example execution is only triggered when you modify one of the examples or the documentation as the execution take a lot of time. You can enable [Github Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository) in your fork to see the result of the CI pipeline. Results are also indicated at the end of your pull request when raised. However note, that workflows in the pull request need approval from the maintainers before being executed.

# Troubleshooting

## Version of pyRiemann not updated
There is a known issue when you install `pyRiemann-qiskit` in an environement where there is already `pyRiemann` installed. In such case, the `pyRiemann` version is not updated. Therefore before installing or updating `pyRiemann-qiskit`, we recommend to install `pyRiemann` as it follows:

```
pip uninstall pyriemann
pip install pyriemann@git+https://github.com/pyRiemann/pyRiemann#egg=pyriemann
```
