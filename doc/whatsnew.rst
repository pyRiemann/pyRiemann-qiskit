.. _whatsnew:

.. currentmodule:: pyriemann-qiskit

What's new in the package
=========================

v0.1.0
------

- Remove support for python 3.7
- Bump dependencies
- Move QuantumClassifierWithDefaultRiemanianPipeline to the pipelines module
- Example using Quantum MDM on real data
- Example with quantum SVM on the titanic dataset
- Example with Motor Imagery
- Multiclass classification
- Add visualization module
- Example with quantum art visualization

Details:

https://github.com/pyRiemann/pyRiemann-qiskit/releases/tag/v0.1.0

v0.0.4
------

- Improve documentation, power point presentation, wiki and Readme
- Bump dependencies
- Fix firebase admin could not load because of google cloud
- Update docker image, and publish them on release
- Add support functions and example for MOABB with firebase connector
- Add a module to regroup quantum provider util functions
- Change quantum simulator to Aer - compatible with CUDA acceleration on Linux.
- Implement convex distance for Quantic MDM (Experimental)
- Improve workflow for Ci/Cd (cache results, automate linting)
- Add regularization of convex mean
- Fix examples not running on Ci/Cd

Details:

https://github.com/pyRiemann/pyRiemann-qiskit/releases/tag/v0.0.4

v0.0.3
------

- Enable python 3.9
- Bump cvxpy and qiskit-ibmq-provider
- Expose C and max_iter parameters for QSVC, SVC and Pegasos QSVC
- Add support for Firebase
- Improve Docker support
- Fix deprecated api method in sphinx
- Improve documentation:

  - display of sphinx documentation
  - Readme
  - `Wiki <https://github.com/pyRiemann/pyRiemann-qiskit/wiki>`_
  - Draft paper of the software

Details:

https://github.com/pyRiemann/pyRiemann-qiskit/releases/tag/v0.0.3

v0.0.2
------

- Migrate from qiskit-aqua to qiskit-ml
- Better support for docplex convex optimization model
- Add support for docker, making possible to use a containerized environment with this project
- Support for Pegasos implementation of quantum support-vector machines

Details:

https://github.com/pyRiemann/pyRiemann-qiskit/releases/tag/v0.0.2-dev

v0.0.1
------

- Repository base architecture
- Qiskit wrapper
- Example with toys dataset and ERP
- Exposure of hyperparameters (Shots, feature map, gamma, optimizer and variational form)
- Support for pytest class and parametrization
- Naive dimension reduction technics
- Default pipeline with Riemann geometry and Qiskit
- Support for docplex model for convex optimization
- Example with scikit-learn GridSearchCV
- Example with MOABB

Details:

https://github.com/pyRiemann/pyRiemann-qiskit/releases/tag/v0.0.1-dev
