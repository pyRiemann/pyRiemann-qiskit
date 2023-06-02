.. _installing:

Installing pyRiemann-qiskit
===========================

There is no yet stable version of pyRiemann-qiskit.

Therefore, it is recommanded to clone the source code on `github <https://github.com/pyRiemann/pyRiemann-qiskit>`__ and install directly the package from source.

``pip install -e .``

The install script will install the required dependencies. If you want also to build the documentation and to run the test locally, you could install all development dependencies with

``pip install -e .[docs,tests]``

If you use a zsh shell, you need to write `pip install -e .\[docs,tests\]`. If you do not know what zsh is, you could use the above command.

You may check that the package was correctly installed by starting a python shell and writing:

``import pyriemann_qiskit``

Dependencies
~~~~~~~~~~~~

-  Python (>= 3.6)

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `cython <https://cython.org/>`__

-  `pyriemann <https://github.com/pyRiemann/pyRiemann-qiskit>`__

-  `qiskit==0.20.0 <https://qiskit.org/>`__

-  `cvxpy=1.1.12 <https://www.cvxpy.org/>`__

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^
These dependencies are recommanded to use the plotting functions of pyriemann or to run examples and tutorials, but they are not mandatory:

-  `matplotlib>=2.2 <https://matplotlib.org/>`__

-  `mne-python <http://mne-tools.github.io/>`__

-  `seaborn <https://seaborn.pydata.org>`__
