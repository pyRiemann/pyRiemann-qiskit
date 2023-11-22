.. _introduction:

Introduction to pyRiemann-qiskit
================================

Litterature on quantum computing suggests it may offer an advantage as compared
with classical computing in terms of computational time and outcomes, such as
for pattern recognition or when using limited training sets [1]_ [2]_.

A ubiquitous library on quantum computing is Qiskit [3]_.
Qiskit is an IBM library distributed under Apache 2.0 which provides both
quantum algorithms and backends. A backend can be either your local machine
or a remote machine, which one can emulates or be a quantum machine.
Qiskit abstraction over the type of machine you want to use, make designing
quantum algorithm seamless.

Qiskit implements a quantum version of support vector
-like classifiers, known as quantum-enhanced support vector classifier (QSVC)
and varitional quantum classifier (VQC) [4]_. These classifiers likely offer
an advantage over classical SVM in situations where the classification task
is complex. Task complexity is raised by the encoding of the data into a
quantum state, the number of available data and the quality of the data. An initial
study is available in [5]_, and it can be downloaded from `here
<https://github.com/pyRiemann/pyRiemann-qiskit/blob/main/doc/Presentations/QuantumERPClassification.pdf>`_.
Although there is no study on this topic at the time of writting,
this could be an interesting research direction to investigate BCI illiteracy.

pyRiemann-qiskit implements a wrapper around QSVC and VQC, to use quantum
classification with Riemannian geometry. A use case would be to use vectorized
covariance matrices in the TangentSpace as an input for these classifiers,
enabling a possible sandbox for researchers and engineers in the field.
`pyRiemann-qiskit` also introduces a quantum version of the famous MDM algorithm.

Quantum drawbacks
================================

- Limitation of the feature dimension

    The number of qubits (and therefore the feature dimension) is limited to:

    - ~36 (depends on system memory size) on a local quantum simulator, and up to:
    - 5000 on a remote quantum simulator;
    - 7 on free real quantum computers, and up to:
    - 127 on exploratory quantum computers (not available for public use).

- Time complexity

    A higher number of trials or dimension increases time to completion of the quantum algorithm, especially when running on a local machine. This is why the number of trials is limited in the examples we provided. However, you should avoid such practices in your own analysis.

    Although these aspects are less important in a remote backend, it may happen that the quantum algorithm is queued depending on the number of concurrent users.

    For all these aspects, the use of pyRiemann-qiskit should be limited to offline analysis only.

References
================================

.. [1] A. Blance and M. Spannowsky,
    ‘Quantum machine learning for particle physics using a variational quantum classifier’,
    J. High Energ. Phys., vol. 2021, no. 2, p. 212, Feb. 2021,
    doi: 10.1007/JHEP02(2021)212.

.. [2] P. Rebentrost, M. Mohseni, and S. Lloyd,
    ‘Quantum Support Vector Machine for Big Data Classification’,
    Phys. Rev. Lett., vol. 113, no. 13, p. 130503, Sep. 2014,
    doi: 10.1103/PhysRevLett.113.130503.

.. [3] H. Abraham et al., Qiskit: An Open-source Framework for Quantum Computing.
    Zenodo, 2019. doi: 10.5281/zenodo.2562110.

.. [4] V. Havlíček et al.,
    ‘Supervised learning with quantum-enhanced feature spaces’,
    Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
    doi: 10.1038/s41586-019-0980-2.

.. [5] G. Cattan, A. Andreev,
    First steps to the classification of ERPs using quantum computation,
    NTB Berlin 2022 - International Forum on Neural Engineering & Brain Technologies, May 2022, Berlin, Germany,
    hal: https://hal.archives-ouvertes.fr/hal-03672246/


How to cite?
================================
Anton Andreev, Grégoire Cattan, Sylvain Chevallier, and Quentin Barthélemy. ‘PyRiemann-Qiskit: A Sandbox for Quantum Classification Experiments with Riemannian Geometry’. Research Ideas and Outcomes 9 (20 March 2023). https://doi.org/10.3897/rio.9.e101006.
