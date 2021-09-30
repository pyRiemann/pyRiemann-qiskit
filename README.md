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
quantum state, the number of available data and the quality of the data.
Although there is no study on this topic at the time of writting,
this could be an interesting research direction to investigate illiteracy
in the domain of brain computer interfaces.

pyRiemann-qiskit implements a wrapper around QSVC and VQC, to use quantum
classification with riemanian geometry. A use case would be to use vectorized
covariance matrices in the TangentSpace as an input for these classifiers,
enabling a possible sandbox for researchers and engineers in the field.

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

## Installation

As there is no stable version, you should clone this repository
and install the package on your local machine using the `setup.py` script

```
python setup.py develop
```

To check the installation, open a python shell and type:

```
import pyriemann_qiskit
```

Full documentation, including API description, is available at <https://pyriemann-qiskit.readthedocs.io/>