.. raw:: html

   <style type="text/css">
   .thumbnail {{
       position: relative;
       float: left;
       margin: 10px;
       width: 180px;
       height: 200px;
   }}

   .thumbnail img {{
       position: absolute;
       display: inline;
       left: 0;
       width: 170px;
       height: 170px;
   }}

   </style>

pyRiemann-qiskit: Qiskit wrapper for pyRiemann
=============================================================

.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
       <a href="auto_examples/ERP/plot_classify_EEG_quantum.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_single_thumb.png">
         </div>
       </a>
       <a href="auto_examples/toys_dataset/plot_classifier_comparison.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_classify_MEG_mdm_thumb.png">
         </div>
       </a>
     </div>
   </div>

   <br>

   <div class="container-fluid">
     <div class="row">
       <div class="col-md-9">

pyRiemann-qiskit is a Qiskit wrapper around pyRiemann. 
It allows to use quantum classification with Riemannian geometry. 
pyRiemann-qiskit provides through Qiskit:
- a sandbox to experiments quantum computing;
- a way to leverage situations where classical computing failed
 or terminates in exponential times;
- an abstraction layer on the quantum backend:
  run your algorithm on either your local or a real quantum machine;
- a way to encode EEG/MEG data into a quantum state;
- a way to measure quantum bit into classical bits.


A typical use case would be to use vectorized covariance matrices in
TangentSpace as an input for quantum classifiers.

For a brief introduction to the ideas behind the package, you can read the
:ref:`introductory notes <introduction>`. More practical information is on the
:ref:`installation page <installing>`. You may also want to browse the
`example gallery <auto_examples/index.html>`_ to get a sense for what you can do with pyRiemann-qiskit
and :ref:`API reference <api_ref>` to find out how.

To see the code or report a bug, please visit the `github repository
<https://github.com/pyRiemann/pyRiemann-qiskit>`_.

.. raw:: html

       </div>
       <div class="col-md-3">
         <div class="panel panel-default">
           <div class="panel-heading">
             <h3 class="panel-title">Content</h3>
           </div>
       <div class="panel-body">

.. toctree::
   :maxdepth: 1

   Introduction <introduction>
   Release notes <whatsnew>
   Installing <installing>
   Example gallery <auto_examples/index>
   API reference <api>

.. raw:: html

       </div>
     </div>
   </div>

   </div>
   </div>
