.. _installing:

Installing pyRiemann-qiskit
===========================

_We recommend the use of [Anaconda](https://www.anaconda.com/) to manage python
environements._

`pyRiemann-qiskit` currently supports Windows, Mac and Linux OS with Python 3.9 - 3.11.

You can install `pyRiemann-qiskit` release from PyPI:

```
pip install pyriemann-qiskit
```

The development version can be installed by cloning this repository and installing the
package on your local machine using the `setup.py` script:

```
python setup.py develop
```

Or directly pip:

```
pip install .
```

Note that the steps above need to be re-executed in your local environment after any
changes inside your local copy of the `pyriemann_qiskit` folder, including pulling from
remote.

To check the installation, open a python shell and type:

```
import pyriemann_qiskit
```

Dependencies
~~~~~~~~~~~~

-  Python (>= 3.9)

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

The project relies mainly on:

- `pyRiemann==0.5.0 <https://github.com/pyRiemann/pyRiemann>`

- `qiskit==1.* <https://qiskit.org/>`__

Other dependencies can be checked in the `setup.py` file.
