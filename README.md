# pyRiemann-qiskit

Litterature on quantum computing suggests it may offer an advantage as compared
with classical computing in terms of computational time and outcomes, such as
for pattern recognition or when using limited training sets [1, 2].

A ubiquitous library on quantum computing is Qiskit [3].
Qiskit is an IBM library distributed under Apache 2.0 which provides both
quantum algorithms and backends. A backend can be either your local machine
or a remote machine, which one can emulates or be a quantum machine.
Qiskit abstraction over the type of machine you want to use, make designing
quantum algorithm seamless.

Qiskit implements a quantum version of support vector
-like classifiers, known as quantum-enhanced support vector classifier (QSVC)
and varitional quantum classifier (VQC) [4]. These classifiers likely offer
an advantage over classical SVM in situations where the classification task
is complex. Task complexity is raised by the encoding of the data into a
quantum state, the number of available data and the quality of the data.
Although there is no study on this topic at the time of writting,
this could be an interesting research direction to investigate illiteracy
in the domain of brain computer interfaces.

pyRiemann-qiskit implements a wrapper around QSVC and VQC, to use quantum
classification with Riemannian geometry. A use case would be to use vectorized
covariance matrices in the TangentSpace as an input for these classifiers,
enabling a possible sandbox for researchers and engineers in the field.

## Quantum drawbacks

- Limitation of the feature dimension

    The number of qubits (and therefore the feature dimension) is limited to:
    - 24 on a local quantum simulator, and up to:
    - 5000 on a remote quantum simulator;
    - 5 on free real quantum computers, and up to:
    - 65 on exploratory quantum computers (not available for public use).

- Time complexity

    A higher number of trials or dimension increases time to completion of the quantum algorithm, especially when running on a local machine. This is why the number of trials is limited in the examples we provided. However, you should avoid such practices in your own analysis. 
    
    Although these aspects are less important in a remote backend, it may happen that the quantum algorithm is queued depending on the number of concurrent users.

    For all these aspects, the use of pyRiemann-qiskit should be limited to offline analysis only.
    
## References

[1] A. Blance and M. Spannowsky,
    ‘Quantum machine learning for particle physics using a variational quantum classifier’,
    J. High Energ. Phys., vol. 2021, no. 2, p. 212, Feb. 2021,
    doi: 10.1007/JHEP02(2021)212.

[2] P. Rebentrost, M. Mohseni, and S. Lloyd,
   ‘Quantum Support Vector Machine for Big Data Classification’,
    Phys. Rev. Lett., vol. 113, no. 13, p. 130503, Sep. 2014,
    doi: 10.1103/PhysRevLett.113.130503.

[3] H. Abraham et al., Qiskit: An Open-source Framework for Quantum Computing.
    Zenodo, 2019. doi: 10.5281/zenodo.2562110.

[4] V. Havlíček et al.,
    ‘Supervised learning with quantum-enhanced feature spaces’,
    Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
    doi: 10.1038/s41586-019-0980-2.

## Installation

_We recommend the use of [Anaconda](https://www.anaconda.com/) to manage python environements._ 

As there is no stable version, you should clone this repository
and install the package on your local machine using the `setup.py` script

```
python setup.py develop
```

To check the installation, open a python shell and type:

```
import pyriemann_qiskit
```

Full documentation, including API description, is available at <https://pyriemann-qiskit.readthedocs.io/>

To run a specific example on your local machine, you should install first dependencies for the documentation:

```
pip install .[doc]
```

Then you can run the python example of your choice like:

```
python examples\ERP\classify_P300_bi.py
```

### Installation with docker

We also offer the possibility to set up the dev environment within docker.
To this end, we recommand to use `vscode` with the `Remote Containers` extension
from Microsoft. 

Once the installation is successful, just open the project in `vscode` and enter `F1`.
In the search bar that opens, type `Rebuild and Reopen Container`.

Wait for the container to build, and open a python shell within the container.
Then ensure everything went smoothly by typing:

```
import pyriemann_qiskit
```

## Contributor Guidelines

Everyone is welcomed to contribute to this repository. There are two types of contributions:

- [Raise an issue](https://github.com/pyRiemann/pyRiemann-qiskit/issues/new) on the repository.
Issues can be either a bug report or an enhancement proposal. Note that it is necessary to register on
GitHub before. There is no special template which is expected but, if you raise a defect please  provide as much details as possible.

- [Raise a pull request](https://github.com/pyRiemann/pyRiemann-qiskit/compare). Fork the repository and work on your own branch. Then raise a pull request with your branch against master. As much as possible, we ask you to:
    - avoid merging master into your branch. Always prefer git rebase.
    - always provide full documentation of public method.

Code contribution (pull request) can be either on core functionalities, documentation or automation.

- The core functionalies are based on `Python`, [pyRiemann](https://github.com/pyRiemann/pyRiemann), [Qiskit ML](https://github.com/Qiskit/qiskit-machine-learning) and follow the best practice from [scikit-learn](https://scikit-learn.org/stable/index.html). We use `flake8` for code formatting. `flake8` is installed with the testing dependencies (see below) or can be installed directly from `pip`:

    ```
    pip install flake8
    ```

    To execute `flake8`, just type `flake8` from the root repository, and correct all errors related to your changes.

- The documentation is based on [Sphinx](https://www.sphinx-doc.org/en/master/).
- Automation is based on `GitHub Action` and `pytest`. It consists in two automated workflows for running the example and the tests. To run the tests on your local machine, you should first install the dependencies for testing:

    ```
    pip install .[test]
    ```

    and then type `pytest` from the root repository. You can also specify a file like:

    ```
    pytest tests/test_classification.py 
    ```

    Workflows are automatically triggered when you push a commit. However, the worflow for example execution is only triggered when you modify one of the examples or the documentation as the execution take a lot of time. You can enable [Github Actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository) in your fork to see the result of the CI pipeline. Results are also indicated at the end of your pull request when raised. However note, that workflows in the pull request need approval from the maintainers before being executed.

### Deploy an example on the cloud

When creating an example, your local computer may be limited in terms of ressources to emulate a quantum computer.
Instead, you might want to use a cloud provider to run the example.
Here we will provide the steps with the Google Cloud Plateform (other cloud providers offer similar functionnalities):

1. Create a new branch with you example in your fork repository. Modify the `/Dockerfile` to redirect the `entrypoint` to your example.
Make sure that the `create_docker_image` workflow passed.
2. Open an account on Google Cloud (it required a billing account, but you will not be charged until you upgrade your account).
3. Create a [Cloud Run Service](https://console.cloud.google.com/run/create?project=pyriemann-qiskit) called `pyriemmann-qiskit` (or any project name which is suitable for you).
For the moment use the default configuration, although you may want to already indicate the physical ressources:

![image](https://user-images.githubusercontent.com/6229031/176449146-d3c3da37-0382-46e6-a20b-1b963ce6c12a.png)

4. Create an [artifactory repository](https://console.cloud.google.com/artifacts/create-repo?project=pyriemann-qiskit), following
the `Create a Docker repository in Artifactory` tutorial. Tutorials are displayed in the right side panel of the plateform.
Make sure to indicate `pyriemann-qiskit` as a project (or the one you created instead).
5. Create a new [Cloud Build Trigger](https://console.cloud.google.com/cloud-build/triggers?project=pyriemann-qiskit). Provide the required permissions to install the `Google Build app` on Github. This will allow Google Cloud to build a container image directly from your fork repository. The process is automated and you only need to follow the steps. For more details click [here](https://cloud.google.com/build/docs/automating-builds/build-repos-from-github).
6. Under `Configuration>Type`, select `Dockerfile`. 
7. Under `Configuration>Location`, select `Repository` and type `Dockerfile` in the input box `Dockerfile name`.
8. Under `Configuration>Location` provide a value for the image name.
It should be in the form: `<XXX>-docker.pkg.dev/<name of your cloud run service>/<name of your docker repo>/<custom image name>:$COMMIT_SHA`.
You can copy the first part of this URL (except the image name) from your artifactory settings:

![image](https://user-images.githubusercontent.com/6229031/176449496-daf5f263-3bb9-4eb9-aad3-7bcf289b8f59.png)

9. Validate the trigger, and run it. Check everything passed.
10. Edit the service you created in step `3`, and select a `Container Image URL`. If everything went well,
a new image should have been pushed in your artifact repository. It is also possible to specify a different entrypoint that the one provided in setp `1`

![image](https://user-images.githubusercontent.com/6229031/176448796-8d2472c5-5662-4b69-8d47-c31ebbe9a7e5.png)

11. Validate the service and click on the `Logs` tab to see the output.

# Troubleshooting

## Version of pyRiemann not updated
There is a known issue when you install `pyRiemann-qiskit` in an environement where there is already `pyRiemann` installed. In such case, the `pyRiemann` version is not updated. Therefore before installing or updating `pyRiemann-qiskit`, we recommend to install `pyRiemann` as it follows:

```
pip uninstall pyriemann
pip install pyriemann@git+https://github.com/pyRiemann/pyRiemann#egg=pyriemann
```
