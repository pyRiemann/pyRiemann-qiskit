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
  - name: Anton Andreev
    affiliation: 3
  - name: Grégoire H. Cattan
    orcid: 0000-0002-7515-0690
    affiliation: 1
  - name: Sylvain Chevallier
    affiliation: 2
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


`pyRiemann-qiskit` implements a wrapper around QSVC and VQC, enabling quantum classification for Riemannian Geometry (RG). It facilitates the creation and parametrization of the quantum backend, and is fully compliant with scikit-learn's transformers, estimators and classifiers so it becomes easy to integrate quantum classification in an existing pipeline. It also supports [docplex](http://ibmdecisionoptimization.github.io/docplex-doc/mp/index.html) for the specification of convex optimization problems, with the limitation of using binary and unconstrained variables.
`pyRiemann-qiskit` is built on top of `pyRiemann`, a machine learning library based on RG, thereby enabling the manipulation of covariance matrices (and in a larger extent semi-positive definite matrices) within the Riemannian manifold such as whitenning, channel selection, projection of the matrices into the tangent space of the Riemanian manifold or classification of the matrices based on Riemannian distances to class prototypes.

The library also includes several examples to guide practitioners, as well as a complete test suite. We will briefly describes below some of the functionalities enabled by the software.

## Support for quantum classifiers
The software supports QSVC and VQC classifiers. The first concern regarding quantum classifiers is the encoding of classical data in quantum states. This operation is known as feature mapping. To obtain an advantage over classical computing, feature mapping must implement quantum circuits, which are difficult to emulate on a classical computer.
Feature mapping is common to VQCs and QSVCs. Both are SVM-like classifiers in the sense that they generate a separating hyperplane. The difference between them is that VQCs use a variational quantum circuit (also known as a variational form) for this task, whereas QSVCs use a quantum-enhanced but conventional SVM.
The software also support the Pegasos implementation of QSVC, which offers a provable speed-up as compared to QSVC [@gentinetta_complexity_2022].

The code snippet below demonstrates how to instantiate VQC or QSVC classifier in pyRiemann-Qiskit:

```
vqc = QuanticVQC()
qsvc = QuanticSVM()
pegasos = QuanticSVM(pegasos=True)
svc = QuanticSVM(quantum=False)
```

By default, the backend will be a local quantum simulator. However, it is possible to register on [IBM quantum](https://quantum-computing.ibm.com/) and request a token to use one of the publicly available quantum computers. 
All classifiers accept a `q_account_token` parameter which, if valid, 
allow the selection of an available quantum computer to run the classification.

However, note that, at the time of writting, the number of qubits (and therefore the feature dimension) is limited to:

- 36 on a local quantum simulator, and up to:
- 5000 on a remote quantum simulator;
- 5-7 on free real quantum computers, and up to:
- 127 on exploratory quantum computers (not available for public use).

## Support for convex optimization problem

The MDM algorithm [@barachant_multiclass_2012] consists in finding the minimum distance between a trial and a class prototype before labelling the trial with the prototype which is the closest to the trial. 

`pyRiemann-qiskit` supports docplex for the definition of convex optimization problems. In particular, this is useful in these two situations:
- computing the barycenter of covariance matrices (class prototype), i.e a matrix being at minimum distances of all matrices. 
- determining the class prototype at minimum distance of a trial - that is, a quadratic optimization problem [@zhao_convex_2019]. 

To calculate the mean we need to select a "distance" and provide an optimizer. For the distance we provide a convex model based on frobenius distance (python method fro_mean_convex). For the optimizer pyRiemann-qiskit relies on Qiskit implementation of QAOA (Quantum Approximate Optimization Algorithm). To test the convex model, we also provide a wrapper on a classical optimizer (CobylaOptimizer, also included within Qiskit).

pyRiemann-qiskit uses a wrapper around QAOA optimizer (class NaiveQAOAOptimizer) that rounds covariance matrices to a certain precision and converts each resulting integer to binary. The implementation is based on Qiskit’s IntegerToBinary, a bounded-coefficient encoding method. However, QAOA is limited to te solving of QUBO problems, that is, problems with unconstrained and binary variables only. Binary variables means that a matrix can contain only 0 and 1 as values.

The complexity of the QAOA optimizer raises as a function of the size of covariance matrices and the upper-bound coefficient. The size of the covariance matrices depends on the number of input channels in the input time epoch, as well as the dimension reduction method which is in place. The upper-bound coefficient also has an impact on the final size of the covariance matrices. If all variables inside a matrix are integers that can take only 4 values, they can be represented by only 2 bits. The size of the matrix will only be twice larger. However, a high upper bound implies a higher number of qubits to hold the variables inside a matrix (and therefore the final size of the binary matrix will be impacted).

Here is an example code:

```
metric = {
'mean': "convex",
'distance': "convex"
}

distance_methods["convex"] = lambda A, B: np.linalg.norm(A - B, ord='fro')

clf = make_pipeline(XdawnCovariances(), MDM(metric=metric))
skf = StratifiedKFold(n_splits=5)
n_matrices, n_channels, n_classes = 100, 3, 2
covset = get_covmats(n_matrices, n_channels)
labels = get_labels(n_matrices, n_classes)

score = cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')
assert score.mean() > 0
```

If the MDM method is supplied with "convex" metric it will automatically use the fro_mean_convex method for computing the mean. The default optimizer for the fro_mean_convex method is the Cobyla optimizer.

## Classification of vectorized covariances matrices

To date, the classification of vectorized covariance matrices is the best solution for quantum. It relies on the so-called `TangentSpace` vectorization, which consist in the projection of the covariance matrices into the tangent space of the Riemannian manifold. 
The dimension of the resulting feature can then be reduced using a PCA for example, in order to match the number of available qubits.

The code snippet below demonstrates how we operate a dimension reduction of the epoch using Xdawn [@rivet_optimal_2013], applied the TangentSpace method and then diminish the size of the feature to match the capability of the quantum backend in our quantum classifier (here `QuanticSVM`)

```
pipe = make_pipeline(XdawnCovariances(nfilter=2),
                     TangentSpace(),
                     PCA(),
                     QuanticSVM())
```

For ease of use, the library provides the `QuantumClassifierWithDefaultRiemannianPipeline` class, which operates the pipeline above. 

## Future work

### Direct classification of covariance matrices 

The MDM algorithm is a decision optimization problem that can be solved using Qiskit's QAOA, at condition of 1) it is provided in the form of a docplex model and, 2) it is quadratic, unconstrained and, contains only binary variables.

For instance, MDM based on Log-Euclidian metric has the following expression [@zhao_convex_2019]: 

$$\arg \min w^{T}Dw - 2 Vec(\log Y) D$$

with $\sum w_i = 1, w_i >= 0 \forall i$ and $D=[Vec(\log X_1)...Vec(\log X_N)]$, $Y$ being the trial and $X_i$ a class. The classes are built during training using the mean covariance matrices, and therefore this approach is compatible with the `fro_mean_convex` method previously introduced.

Note that the equation above is a quadratic optimization problem. However, weights in the w vector are constrained continuous variables, thus complicating the use of the IntegerToBinary method.

In addition, the equation must be solved for each new trial that needs to be classified. The complexity of determining the correct weight to minimize the equation varies as a function of the number of classes and the upper bound coefficient which is used for the `IntegerToBinary method` (the higher this coefficient, the higher the complexity). While potentially slower, this quantum-optimized version of the MDM algorithm _could_ produce better results, especially in cases where classical computation fails.

### Multi-class classifications

At the time of writting, `pyRiemann-qiskit` only supports binary classification of covariance matrices. Furthur work also include the implementation of multi-class classification.

# Statement of need

`pyRiemann-qiskit` is a sandbox to test quantum computing with RG. It unifies within a same library quantum and RG tools to seamingless integrate quantum classification with RG, a ubiquitous framework in the domain of Brain-Computer Interfaces (BCI). Therefore, the primary audience we target are practitioners coming from the BCI field, willing to experiment quantum computing for the classification of electroencephalography (EEG) or magnetoencephalography signals. An initial study on this topic is available in [@cattan_first_2022], and it can be downloaded from [pyRiemann-qiskit](https://github.com/pyRiemann/pyRiemann-qiskit/blob/main/doc/Presentations/QuantumERPClassification.pdf). However note that the tools provided by the library are also relevant for others domains, such as classification of other biometrical signals, image recognition or detection of fraud transaction (e.g. @grossi_mixed_2022).

In brief, we hope this library will furthur the acceptance of quantum and RG technologies for concrete applications, opening new and interesting research fields. For example, this could be an interesting research direction to investigate BCI illiteracy, a situation in which classical classifiers usually fail to generalize the appropriate EEG signal from the data.

# References
