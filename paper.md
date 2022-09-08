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

Litterature on quantum computing suggests it may offer an advantage as compared with classical computing in terms of computational time and outcomes, such as for pattern recognition or when using limited training sets [1, 2].

A ubiquitous library on quantum computing is Qiskit [3]. Qiskit is an IBM library distributed under Apache 2.0 which provides both quantum algorithms and backends. A backend can be either your local machine or a remote machine, which one can emulates or be a quantum machine. Qiskit abstraction over the type of machine you want to use, make designing quantum algorithm seamless.

Qiskit implements a quantum version of support vector -like classifiers, known as quantum-enhanced support vector classifier (QSVC) and varitional quantum classifier (VQC) [4]. In practice, experiment on artificial datasets suggests that quantum enhanced SVMs offer a provable speedup compared to classical algorithms [42]. These classifiers likely offer an advantage over classical SVM in situations where the classification task is complex. Task complexity is raised by the encoding of the data into a quantum state, the number of available data and the quality of the data. 


pyRiemann-qiskit implements a wrapper around QSVC and VQC, enabling quantum classification for Riemannian Geometry (RG). It facilitates the creation and parametrization of the quantum backend, and is fully compliant with scikit-learn implementation of transformers and classifiers so it become easy to integrate quantum classification in an existing pipeline. It also support docplex for the specification of convex optimization problem, with the limitation of using binary and unconstrained variables. The library also includes several examples to guide practitioners, and complete test suite. We will briefly describes below some of the functionalities enabled by the software.

## Classification of vectorized covariances matrices with a quantum SVM
Quantum SVM
-> Pegagos implementation
-> Complexity of Pegasos vs Classical SVM
Quantum VQC
-> Link to neural network

## Support for convex optimization problem

QUBO
-> The integer to binary problem
-> not adapted to physical data. 
Docplex model for mean

## Direct classification of covariance matrices 

Docplex model for MDM


Future WORK //TODO

 [@Luck:2012]

# Statement of need

pyRiemann-qiskit is a sandbox to test quantum computing with RG. It unifies within a same library quantum and RG tools to seamingless integrate quantum classification with RG, a ubiquitous framework in the domain of Brain-Computer Interfaces (BCI). Therefore, the primary audience we target are practitioners coming from the BCI field, willing to experiment quantum computing for the classification of electroencephalography (EEG) or magnetoencephalography signals. An initial study on this topic is available in [5], and it can be downloaded from [here]. However note that the tools provided by the library are also relevant for others domains, such as classification of other biometrical signal, image recognition or fraud transaction (e.g. []).

In brief, we hope this library will furthur the acceptance of quantum and RG technologies for concrete applications, opening new and interesting research fields. For example, this could be an interesting research direction to investigate BCI illiteracy, a situation in which classical classifiers usually fails to generalize the appropriate EEG signal from the data.

[@Cattan:2018, an early version of this software was used in @korczowski:2019a; @korczowski:2019b; @korczowski:2019c; @korczowski:2019d; @Vaineau:2019; @VanVeen:2019; @Cattan:2019; @Cattan:2021 thereby outlining the need for such an implementation.]

# References
