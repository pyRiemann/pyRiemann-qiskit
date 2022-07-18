.. _api_ref:

=============
API reference
=============

Classification
--------------
.. _classification_api:
.. currentmodule:: pyriemann_qiskit.classification

.. autosummary::
    :toctree: generated/
    :template: class.rst

    QuanticClassifierBase
    QuanticSVM
    QuanticVQC


Utils function
--------------

Utils functions are low level functions for the `classification` module.

Hyper-parameters generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _hyper_params_factory_api:
.. currentmodule:: pyriemann_qiskit.utils.hyper_params_factory

.. autosummary::
    :toctree: generated/

    gen_zz_feature_map
    gen_two_local
    get_spsa
    get_spsa_parameters

Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _filtering_api:
.. currentmodule:: pyriemann_qiskit.utils.filtering

.. autosummary::
    :toctree: generated/

    NoDimRed
    NaiveDimRed

Docplex
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _docplex_api:
.. currentmodule:: pyriemann_qiskit.utils.docplex

.. autosummary::
    :toctree: generated/

    square_cont_mat_var
    square_int_mat_var
    square_bin_mat_var
    pyQiskitOptimizer
    ClassicalOptimizer
    NaiveQAOAOptimizer

Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _datasets_api:
.. currentmodule:: pyriemann_qiskit.datasets.utils

.. autosummary::
    :toctree: generated/

    get_mne_sample
    get_linearly_separable_dataset
    get_qiskit_dataset
    get_feature_dimension

