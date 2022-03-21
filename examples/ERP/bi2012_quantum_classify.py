"""
====================================================================
Quantum classification of ERPs using data from bi2012
====================================================================

#TODO

"""
# Author: #TODO
# License: BSD (3-clause)

from pyriemann_qiskit.datasets import get_bi2012_dataset

bi2012 = get_bi2012_dataset()
for (X, y) in bi2012:
    assert X is not None and y is not None
    # TODO
