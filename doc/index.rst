.. pyRiemann-qiskit documentation master file

====================================================
pyRiemann-qiskit: Quantum Machine Learning for BCI
====================================================

.. raw:: html

   <div class="hero-section" style="text-align: center; padding: 2em 0; margin-bottom: 2em;">
      <p style="font-size: 1.3em; color: #4c72b0; max-width: 800px; margin: 0 auto; line-height: 1.6;">
         A powerful Qiskit wrapper for pyRiemann that brings quantum computing
         to Riemannian geometry-based brain-computer interfaces.
      </p>
   </div>

.. grid:: 1 1 2 3
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: 🚀 Quick Start
      :link: installing
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

      Get started with pyRiemann-qiskit in minutes. Install and run your first quantum classifier.

   .. grid-item-card:: 📚 Examples
      :link: auto_examples/index
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

      Explore our gallery of examples showcasing quantum classification with EEG/MEG data.

   .. grid-item-card:: 🔧 API Reference
      :link: api
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

      Complete API documentation for all classes and functions.

Overview
========

**pyRiemann-qiskit** is a `Qiskit <https://github.com/Qiskit>`_ wrapper around
`pyRiemann <https://github.com/pyRiemann/pyRiemann>`_ that enables quantum
classification with Riemannian geometry for brain-computer interface applications.

Key Features
------------

✨ **Quantum Algorithms**
   Leverage quantum computing for pattern recognition and classification tasks.

🧠 **Brain Signal Processing**
   Specialized tools for EEG/MEG data analysis using Riemannian geometry.

🔬 **Research Sandbox**
   Experiment with quantum machine learning in a flexible environment.

⚡ **Backend Flexibility**
   Run on local simulators, remote simulators, or real quantum hardware.

🎯 **Quantum Classifiers**
   - Quantum Support Vector Machines (QSVM)
   - Variational Quantum Classifiers (VQC)
   - Quantum Minimum Distance to Mean (MDM)
   - Nearest Convex Hull (NCH) algorithms

Use Cases
---------

A typical workflow involves:

1. **Preprocessing**: Extract covariance matrices from EEG/MEG signals
2. **Tangent Space**: Project matrices to tangent space for vectorization
3. **Quantum Encoding**: Encode features into quantum states
4. **Classification**: Use quantum algorithms for pattern recognition

.. code-block:: python

   from pyriemann_qiskit.classification import QuanticSVM
   from pyriemann.estimation import Covariances
   from pyriemann.tangentspace import TangentSpace
   from sklearn.pipeline import make_pipeline

   # Create a quantum classification pipeline
   clf = make_pipeline(
       Covariances(),
       TangentSpace(),
       QuanticSVM()
   )

   # Train and predict
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)

Featured Examples
=================

.. raw:: html

   <div class="gallery-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2em; margin: 2em 0;">

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card::
      :link: auto_examples/ERP/plot_classify_EEG_quantum_svm
      :link-type: doc
      :img-top: _images/sphx_glr_plot_classify_EEG_quantum_svm_thumb.png
      :class-card: sd-border-0 sd-shadow-md

      **Quantum SVM for EEG**
      ^^^
      Classify EEG signals using quantum support vector machines.

   .. grid-item-card::
      :link: auto_examples/ERP/plot_classify_P300_bi
      :link-type: doc
      :img-top: _images/sphx_glr_plot_classify_P300_bi_thumb.png
      :class-card: sd-border-0 sd-shadow-md

      **P300 Classification**
      ^^^
      Binary classification of P300 event-related potentials.

   .. grid-item-card::
      :link: auto_examples/toys_dataset/plot_classifier_comparison
      :link-type: doc
      :img-top: _images/sphx_glr_plot_classifier_comparison_thumb.png
      :class-card: sd-border-0 sd-shadow-md

      **Classifier Comparison**
      ^^^
      Compare quantum vs classical classifiers on toy datasets.

.. raw:: html

   </div>

Important Considerations
========================

.. admonition:: Quantum Limitations
   :class: warning

   **Feature Dimension Limits**
      - Local simulator: ~36 qubits
      - Real quantum hardware: up to 156 qubits

   **Time Complexity**
      Quantum algorithms may take longer on local machines. Use remote backends
      for better performance. Suitable for offline analysis only.

.. admonition:: Getting Help
   :class: tip

   - 📖 Read the :ref:`introduction <introduction>` for background
   - 💻 Check the :ref:`installation guide <installing>`
   - 🎨 Browse the `example gallery <auto_examples/index.html>`_
   - 🐛 Report bugs on `GitHub <https://github.com/pyRiemann/pyRiemann-qiskit>`_

Citation
========

If you use pyRiemann-qiskit in your research, please cite:

.. code-block:: bibtex

   @article{andreev2023pyriemann,
     title={pyRiemann-qiskit: A Sandbox for Quantum Classification
            Experiments with Riemannian Geometry},
     author={Andreev, Anton and Cattan, Gr{\'e}goire and
             Chevallier, Sylvain and Barth{\'e}lemy, Quentin},
     journal={Research Ideas and Outcomes},
     volume={9},
     year={2023},
     publisher={Pensoft Publishers},
     doi={10.3897/rio.9.e101006}
   }

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installing
   whatsnew

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Made with Bob
