---
title: 'pyRiemann-qiskit: A Sandbox to Experiment Quantum Classification with Riemannian Geometry'
tags:
  - Riemmanian Geometry
  - Quantum computing
  - pyRiemann
  - Qiskit
  - Event-Related Potential (ERP)
  - Brain-Computer Interface (BCI)
authors:
  - name: Grégoire H. Cattan
    orcid: 0000-0002-7515-0690
    affiliation: 1
  - name: Sylvain Chevallier
    affiliation: 2
  - name: Anton Andreev
    affiliation: 3
  - name: Quentin Barthélemy
    affiliation: 4
affiliations:
 - name: IBM Software, Data and AI, Poland
   index: 1
 - name: LISV, University of Paris-Saclay, France
   index: 2
 - name: GIPSA-lab, CNRS, University of Grenoble-Alpes, France
   index: 3
 - name: Foxstream, Vaulx-en-Velin, France
   index: 4
date: 8 September 2022
bibliography: paper.bib

---

# Summary

Litterature on quantum computing suggests it may offer an advantage as compared with classical computing in terms of computational time and outcomes, such as for pattern recognition or when using limited training sets [@blance_quantum_2021; @rebentrost_quantum_2014].

A ubiquitous library on quantum computing is Qiskit [@abraham_qiskit_2019]. Qiskit is an IBM library distributed under Apache 2.0 which provides both quantum algorithms and backends. A backend can be either your local machine or a remote machine, which one can emulates or be a quantum machine. Qiskit abstraction over the type of machine you want to use, make designing quantum algorithm seamless.

Qiskit implements a quantum version of support vector -like classifiers, known as quantum-enhanced support vector classifier (QSVC) and varitional quantum classifier (VQC) [@havlicek_supervised_2019]. In practice, experiment on artificial datasets suggests that quantum enhanced SVMs offer a provable speedup compared to classical algorithms [@liu_rigorous_2021]. These classifiers likely offer an advantage over classical SVM in situations where the classification task is complex. Task complexity is raised by the encoding of the data into a quantum state, the number of available data and the quality of the data. 


pyRiemann-qiskit implements a wrapper around QSVC and VQC, enabling quantum classification for Riemannian Geometry (RG). It facilitates the creation and parametrization of the quantum backend, and is fully compliant with scikit-learn implementation of transformers and classifiers so it become easy to integrate quantum classification in an existing pipeline. It also support docplex [REF] for the specification of convex optimization problem, with the limitation of using binary and unconstrained variables. The library also includes several examples to guide practitioners, as well as a complete test suite. We will briefly describes below some of the functionalities enabled by the software.

## Support for quantum classifiers
The software supports QSVC and VQC classifiers. 
The first concern regarding quantum classifiers is the encoding of classical data in quantum states. This operation is known as feature mapping. To obtain an advantage over classical computing, feature mapping must implement quantum circuits, which are difficult to emulate on a classical computer.
Feature mapping is common to VQCs and QSVCs. Both are SVM-like classifiers in the sense that they generate a separating hyperplane. The difference between them is that VQCs use a variational quantum circuit (also known as a variational form) for this task, whereas QSVCs use a quantum-enhanced but conventional SVM.

The software also support the Pegasos implementation of QSVC, which offers a provable speed-up as compared to QSVC [@gentinetta_complexity_2022].

The code snippet below demonstrates how to instantiate VQC or QSVC classifier in pyRiemann-Qiskit:

```
vqc = QuanticVQC()
qsvc = QuanticSVM()
pegasos = QuanticSVM(pegasos=True)
svc = QuanticSVM(quantum=False)
```

By default, the backend will be a local quantum simulator. However, it is possible to register on [IBM quantum](https://quantum-computing.ibm.com/) and request a token to use one of the publicly available quantum computer. 
All classifiers accept a `q_account_token` parameters which, if valid, 
will select an available quantum computer to run the classification.

However, note that, at the time of writting, the number of qubits (and therefore the feature dimension) is limited to:

- 36 on a local quantum simulator, and up to:
- 5000 on a remote quantum simulator;
- 5-7 on free real quantum computers, and up to:
- 127 on exploratory quantum computers (not available for public use).

## Support for convex optimization problem

`pyRiemann-qiskit` supports [docplex](http://ibmdecisionoptimization.github.io/docplex-doc/mp/index.html) for the definition of convex optimization problem. In particular, this is usefull in these two situations:
- computing the barycenter of covariance matrices, i.e the covariance matrix which is at minimum distances of all inputs. 
- determing the class prototype which is at minimum distance of a trial, which we can define as a quadratic optimization problem [e.g. ]. 

The library relies on `Qiskit` implementation of `QAOA` (Quantum Approximate Optimization Algorithm) which is limited to the solving of
QUBO problems, that is, problems with unconstrained and binary variables only.

We provide a convex model for the first situation based on frobenius distance (method `fro_mean_convex`), as well as a wrapper around QAOA optimizer (class `NaiveQAOAOptimizer`) that round covariance matrices to a certain precision and convert each resulting integer to binary. 
The implementation is based on Qiskit's `IntegerToBinary`, a bounded-coefficient encoding method.

Complexity of the optimizer raise as a function of the matrix size and the bound coefficient, and hence it best adapt to covariance matrices having a limited number of channels, with naturally bounded and "differentiable" values.


## Classification of vectorized covariances matrices

To date, the classification of vectorized covariance matrices is the best solution for quantum. It relies on the so-called `TangentSpace` vectorization, which consist in the projection of the covariance matrices into the tangent spaces of the riemannian manifold. 
The dimension of the resulting feature can then be reduced using a PCA for example, in order to match the number of available qbits.

The code snippet below demonstrate how we operate a dimension reduction of the epoch using Xdawn, applied the TangentSpace method and then diminish the feature of the feature to match the capability of the quantum backend in our quantum classifier (QuanticSVM)

```
pipe = make_pipeline(XdawnCovariances(nfilter=2),
                     TangentSpace(),
                     PCA(),
                     QuanticSVM())
```

For ease of use, the library provide the `QuantumClassifierWithDefaultRiemannianPipeline` class, which operates the pipeline above. 

## Future work

### Direct classification of covariance matrices 

The MDM algorithm consists in finding the minimum distance between a trial and a class prototype and labelling the trial with the prototype which is the closest to the trial. 
It is a decision optimization problem that can be solved using Qiskit's QAOA, at condition of 1) it is provided in the form of a docplex model, and 2) it is quadratic, unconstrained, and contains only binary variables.

The MDM algoritm, based on Log-Euclidian metric for example has the following expression: 
arg min w w> Dw − 2 Vec(logY) D , s.t. e>w = 1, w ≥ 0 , (22) where D = [Vec(log X1) · · · Vec(log XN)], e = [1 · · · 1]>.

X are the class prototypes, Y is a trial

We know it is a quadratic optimisation problem, however w is a vector that contains only continuous variables. Weights admit a lower bound, but no upper-bound, therefore utilization of the IntegerToBinary method is limited in practice. 

### Multi-class classifications

`pyRiemann-qiskit` only support binary classification for the moment. Futur work on the library include the implementation of `multi-class` classification.

# Statement of need

`pyRiemann-qiskit` is a sandbox to test quantum computing with RG. It unifies within a same library quantum and RG tools to seamingless integrate quantum classification with RG, a ubiquitous framework in the domain of Brain-Computer Interfaces (BCI). Therefore, the primary audience we target are practitioners coming from the BCI field, willing to experiment quantum computing for the classification of electroencephalography (EEG) or magnetoencephalography signals. An initial study on this topic is available in [@cattan_first_2022], and it can be downloaded from [pyRiemann-qiskit](https://github.com/pyRiemann/pyRiemann-qiskit/blob/main/doc/Presentations/QuantumERPClassification.pdf). However note that the tools provided by the library are also relevant for others domains, such as classification of other biometrical signals, image recognition or detection of fraud transaction (e.g. @grossi_mixed_2022).

In brief, we hope this library will furthur the acceptance of quantum and RG technologies for concrete applications, opening new and interesting research fields. For example, this could be an interesting research direction to investigate BCI illiteracy, a situation in which classical classifiers usually fails to generalize the appropriate EEG signal from the data.

# References
