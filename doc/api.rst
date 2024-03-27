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
    QuanticMDM


Pipelines
---------
.. _pipelines_api:
.. currentmodule:: pyriemann_qiskit.pipelines

.. autosummary::
    :toctree: generated/
    :template: class.rst

    BasePipeline
    QuantumClassifierWithDefaultRiemannianPipeline
    QuantumMDMWithRiemannianPipeline
    QuantumMDMVotingClassifier


Ensemble
---------
.. _ensemble_api:
.. currentmodule:: pyriemann_qiskit.ensemble

.. autosummary::
    :toctree: generated/
    :template: class.rst

    JudgeClassifier


Utils functions
----------------

Utils functions are low level functions for the `classification` and `pipelines` module.

Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _utils_api:
.. currentmodule:: pyriemann_qiskit.utils.utils

.. autosummary::
    :toctree: generated/

    is_qfunction

Hyper-parameters generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _hyper_params_factory_api:
.. currentmodule:: pyriemann_qiskit.utils.hyper_params_factory

.. autosummary::
    :toctree: generated/

    gen_x_feature_map
    gen_z_feature_map
    gen_zz_feature_map
    gen_two_local
    get_spsa
    get_spsa_parameters

Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _preprocessing_api:
.. currentmodule:: pyriemann_qiskit.utils.preprocessing

.. autosummary::
    :toctree: generated/

    NdRobustScaler

Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _filtering_api:
.. currentmodule:: pyriemann_qiskit.utils.filtering

.. autosummary::
    :toctree: generated/

    NoDimRed
    NaiveDimRed
    Vectorizer

Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _mean_api:
.. currentmodule:: pyriemann_qiskit.utils.mean

.. autosummary::
    :toctree: generated/

    qmean_euclid
    qmean_logeuclid

Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _distance_api:
.. currentmodule:: pyriemann_qiskit.utils.distance

.. autosummary::
    :toctree: generated/

    qdistance_logeuclid_to_convex_hull
    weights_logeuclid_to_convex_hull

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
    set_global_optimizer
    get_global_optimizer

Math
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _math_api:
.. currentmodule:: pyriemann_qiskit.utils.math

.. autosummary::
    :toctree: generated/

    cov_to_corr_matrix
    union_of_diff
    to_xyz

Firebase
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _firebase_api:
.. currentmodule:: pyriemann_qiskit.utils.firebase_connector

.. autosummary::
    :toctree: generated/

    FirebaseConnector
    Cache
    generate_caches
    filter_subjects_by_incomplete_results
    add_moabb_dataframe_results_to_caches
    convert_caches_to_dataframes

Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _datasets_api:
.. currentmodule:: pyriemann_qiskit.datasets.utils

.. autosummary::
    :toctree: generated/

    get_mne_sample
    generate_linearly_separable_dataset
    generate_qiskit_dataset
    get_feature_dimension
    MockDataset

Quantum Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _quantum_provider_api:
.. currentmodule:: pyriemann_qiskit.utils.quantum_provider

.. autosummary::
    :toctree: generated/

    get_provider
    get_devices
    get_simulator


Visualization
--------------

Helpers to visualize distances, manifold and even "artistic" representation.

Art
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _art_api:
.. currentmodule:: pyriemann_qiskit.visualization.art

.. autosummary::
    :toctree: generated/

    weights_spiral

Manifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _manifold_api:
.. currentmodule:: pyriemann_qiskit.visualization.manifold

.. autosummary::
    :toctree: generated/

    plot_cvx_hull
    plot_manifold
