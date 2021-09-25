"""Module for classification function."""
import numpy as np

from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from .utils.mean import mean_covariance
from .utils.distance import distance
from .tangentspace import FGDA, TangentSpace

from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.quantum_instance import logger
from qiskit.aqua.algorithms import QSVM, SklearnSVM, VQC
from qiskit.aqua.utils import get_feature_dimension
from qiskit.providers.ibmq import least_busy
from qiskit.aqua.components.optimizers import SPSA
from datetime import datetime
import logging
logger.level = logging.INFO


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    """

    def __init__(self, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.metric = metric
        self.n_jobs = n_jobs

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if self.n_jobs == 1:
            self.covmeans_ = [
                mean_covariance(X[y == ll], metric=self.metric_mean,
                                sample_weight=sample_weight[y == ll])
                for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == ll], metric=self.metric_mean,
                                         sample_weight=sample_weight[y == ll])
                for ll in self.classes_)

        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.covmeans_)

        if self.n_jobs == 1:
            dist = [distance(covtest, self.covmeans_[m], self.metric_dist)
                    for m in range(Nc)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                covtest, self.covmeans_[m], self.metric_dist)
                for m in range(Nc))

        dist = np.concatenate(dist, axis=1)
        return dist

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(X)**2)


class FgMDM(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1]_, and classify using MDM.
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular MDM.
    This is basically a pipeline of FGDA and MDM.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    .. [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian
        geometry applied to BCI classification", 9th International Conference
        Latent Variable Analysis and Signal Separation (LVA/ICA 2010),
        LNCS vol. 6365, 2010, p. 629-636.

    .. [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification
        of covariance matrices using a Riemannian-based kernel for BCI
        applications", in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False, n_jobs=1):
        """Init."""
        self.metric = metric
        self.n_jobs = n_jobs
        self.tsupdate = tsupdate

        if isinstance(metric, str):
            self.metric_mean = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self.classes_ = np.unique(y)
        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA(metric=self.metric_mean, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
        self.classes_ = self._mdm.classes_
        return self

    def predict(self, X):
        """get the predictions after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def predict_proba(self, X):
        """Predict proba using softmax after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)

    def transform(self, X):
        """get the distance to each centroid after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class TSclassifier(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.

    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space

    See Also
    --------
    TangentSpace

    Notes
    -----
    .. versionadded:: 0.2.4
    """

    def __init__(self, metric='riemann', tsupdate=False,
                 clf=LogisticRegression()):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf

        if not isinstance(clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')

    def fit(self, X, y):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        self.classes_ = np.unique(y)
        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


class KNearestNeighbor(MDM):

    """Classification by K-NearestNeighbor.

    Classification by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    class is affected according to the majority class of the k nearest
    neighbors.

    Parameters
    ----------
    n_neighbors : int, (default: 5)
        Number of neighbors.
    metric : string | dict (default: 'riemann')
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the distance to the training set in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    MDM

    """

    def __init__(self, n_neighbors=5, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        MDM.__init__(self, metric=metric, n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.classes_ = y
        self.covmeans_ = X

        return self

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_classes = self.classes_[np.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()


class QuanticBase(BaseEstimator, ClassifierMixin):

    """Quantum classification.

    This class implements a SKLearn wrapper around Qiskit library.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:
    - There is no noise on a simulated quantum computer (so results are better)
    - Real quantum computer are quicker than simulator
    - Real quantum computer tasks are assigned to a queue
      before being executed on a back-end

    WARNING: At the moment this implementation only supports binary
    classification (eg. Target vs Non-Target experiment)

    Parameters
    ----------
    target : int
        Label of the target symbol
    qAccountToken : string (default:None)
        If quantum==True and qAccountToken provided,
        the classification task will be running on a IBM quantum backend
    processVector : lambda vector: processedVector (default)
        Additional processing on the input vectors. eg: downsampling
    verbose : bool (default:True)
        If true will output all intermediate results and logs
    quantum : Bool (default:True)
        - If true will run on local or remote backend
        (depending on qAccountToken value).
        - If false, will perform classical computing instead
    **parameters : dictionnary
        This is used by  SKLearn with get_params and set_params method
        in order to create a deepcopy of an instance

    Attributes
    ----------
    classes_ : list
        list of classes.
    verbose : see above
    processVector : see above
    qAccountToken : see above
    target : see above
    quantum : see above
    test_input : Dictionnary
        Contains vectorized test set for target and non-target classes
    training_input : Dictionnary
        Contains vectorized training set for target and non-target classes
    provider : IBMQ Provider
        This service provide a remote quantum computer backend
    backend : Quantum computer or simulator
    feature_dim : int
        Size of the vectorized matrix which is passed to quantum classifier
    new_feature_dim : int
        Feature dimension after proccessed by `processVector` lambda
    prev_fit_params : Dictionnary of data and labels
        Keep in memory data and labels passed to fit method.
        This is used for self-calibration.
    feature_map: ZZFeatureMap
        Transform data into quantum space
    quantum_instance: QuantumInstance (Object)
        Backend with specific parameters (number of shots, etc.)

    See Also
    --------
    QuanticSVM
    QuanticVQC

    """

    def __init__(self, target, qAccountToken=None, quantum=True,
                 processVector=lambda v: v, verbose=True, **parameters):
        self.verbose = verbose
        self.log("Initializing Quantum Classifier")
        self.test_input = {}
        self.set_params(**parameters)
        self.processVector = processVector
        self.qAccountToken = qAccountToken
        self.training_input = {}
        self.target = target
        self.quantum = quantum
        if quantum:
            aqua_globals.random_seed = datetime.now().microsecond
            self.log("seed = ", aqua_globals.random_seed)
            if qAccountToken:
                self.log("Real quantum computation will be performed")
                IBMQ.delete_account()
                IBMQ.save_account(qAccountToken)
                IBMQ.load_account()
                self.log("Getting provider...")
                self.provider = IBMQ.get_provider(hub='ibm-q')
            else:
                self.log("Quantum simulation will be performed")
                self.backend = BasicAer.get_backend('qasm_simulator')
        else:
            self.log("Classical SVM will be performed")

    def log(self, *values):
        if self.verbose:
            print("[QClass] ", *values)

    def vectorize(self, X):
        vector = X.reshape(len(X), self.feature_dim)
        return [self.processVector(x) for x in vector]

    def splitTargetAndNonTarget(self, X, y):
        self.log("""[Warning] Spitting target from non target.
                 Only binary classification is supported.""")
        nbSensor = len(X[0])
        try:
            nbSamples = len(X[0][0])
        except Exception:
            nbSamples = 1
        self.feature_dim = nbSensor * nbSamples
        self.log("Feature dimension = ", self.feature_dim)
        Xta = X[y == self.target]
        Xnt = X[np.logical_not(y == self.target)]
        VectorizedXta = self.vectorize(Xta)
        VectorizedXnt = self.vectorize(Xnt)
        self.new_feature_dim = len(VectorizedXta[0])
        self.log("Feature dimension after vector processing = ",
                 self.new_feature_dim)
        return (VectorizedXta, VectorizedXnt)

    def additionnal_setup(self):
        self.log("There is no additional setup.")

    def fit(self, X, y):
        self.log("Fitting: ", X.shape)
        self.prev_fit_params = {"X": X, "y": y}
        self.classes_ = np.unique(y)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)

        self.training_input["Target"] = VectorizedXta
        self.training_input["NonTarget"] = VectorizedXnt
        self.log(get_feature_dimension(self.training_input))
        feature_dim = get_feature_dimension(self.training_input)
        self.feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2,
                                        entanglement='linear')
        self.additionnal_setup()
        if self.quantum:
            if not hasattr(self, "backend"):
                def filters(device):
                    return (
                        device.configuration().n_qubits >= self.new_feature_dim
                        and not device.configuration().simulator
                        and device.status().operational)
                devices = self.provider.backends(filters=filters)
                try:
                    self.backend = least_busy(devices)
                except Exception:
                    self.log("Devices are all busy. Getting the first one...")
                    self.backend = devices[0]
                self.log("Quantum backend = ", self.backend)
            seed_sim = aqua_globals.random_seed
            seed_trans = aqua_globals.random_seed
            self.quantum_instance = QuantumInstance(self.backend, shots=1024,
                                                    seed_simulator=seed_sim,
                                                    seed_transpiler=seed_trans)
        return self

    def get_params(self, deep=True):
        # Class is re-instanciated for each fold of a cv pipeline.
        # Deep copy of the original instance is insure trough this method
        # and the pending one set_params
        return {
            "target": self.target,
            "qAccountToken": self.qAccountToken,
            "quantum": self.quantum,
            "processVector": self.processVector,
            "verbose": self.verbose,
            "test_input": self.test_input,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def run(self, predict_set=None):
        raise Exception("Run method was not implemented")

    def self_calibration(self):
        X = self.prev_fit_params["X"]
        y = self.prev_fit_params["y"]
        test_per = 0.33
        self.log("Test size = ", test_per, " of previous fitting.")
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_per)
        self.fit(X_train, y_train)
        self.score(X_test, y_test)

    def predict(self, X):
        if(len(self.test_input) == 0):
            self.log("There is no test inputs. Self-calibrating...")
            self.self_calibration()
        result = None
        predict_set = self.vectorize(X)
        self.log("Prediction: ", X.shape)
        result = self.run(predict_set)
        self.log("Prediction finished. Returning predicted labels")
        return result["predicted_labels"]

    def predict_proba(self, X):
        self.log("""[WARNING] SVM prediction probabilities are not available.
                 Results from predict will be used instead.""")
        predicted_labels = self.predict(X)
        ret = [np.array([c == 0, c == 1]) for c in predicted_labels]
        return np.array(ret)

    def score(self, X, y):
        self.log("Scoring: ", X.shape)
        VectorizedXta, VectorizedXnt = self.splitTargetAndNonTarget(X, y)
        self.test_input = {}
        self.test_input["Target"] = VectorizedXta
        self.test_input["NonTarget"] = VectorizedXnt
        result = self.run()
        balanced_accuracy = result["testing_accuracy"]
        self.log("Balanced accuracy = ", balanced_accuracy)
        return balanced_accuracy


class QuanticSVM(QuanticBase):

    """Quantum-enhanced SVM classification.

    This class implements SVC on a quantum machine.
    Note if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    See Also
    --------
    QuanticBase

    """

    def run(self, predict_set=None):
        self.log("SVM classification running...")
        if self.quantum:
            self.log("Quantum instance is ", self.quantum_instance)
            qsvm = QSVM(self.feature_map, self.training_input,
                        self.test_input, predict_set)
            result = qsvm.run(self.quantum_instance)
        else:
            result = SklearnSVM(self.training_input,
                                self.test_input, predict_set).run()
        self.log(result)
        return result


class QuanticVQC(QuanticBase):

    """Variational Quantum Classifier

    Note there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not)

    Parameters
    ----------
    target : see QuanticBase
    qAccountToken : see QuanticBase
    processVector : see QuanticBase
    verbose : see QuanticBase
    parameters : see QuanticBase

    Attributes
    ----------
    optimizer: SPSA
        SPSA is a descent method capable of finding global minima
        https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.SPSA.html
    var_form: TwoLocal
        In quantum mechanics, the variational method is one way of finding
        approximations to the lowest energy eigenstate
        https://qiskit.org/documentation/apidoc/qiskit.aqua.components.variational_forms.html

    See Also
    --------
    QuanticBase

    """

    def __init__(self, target, qAccountToken=None,
                 processVector=lambda v: v, verbose=True, **parameters):
        QuanticBase.__init__(self, target=target, qAccountToken=qAccountToken,
                             processVector=processVector, verbose=verbose,
                             **parameters)

    def additionnal_setup(self):
        self.optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
        self.var_form = TwoLocal(self.new_feature_dim,
                                 ['ry', 'rz'], 'cz', reps=3)

    def run(self, predict_set=None):
        self.log("VQC classification running...")
        vqc = VQC(self.optimizer, self.feature_map, self.var_form,
                  self.training_input, self.test_input, predict_set)
        result = vqc.run(self.quantum_instance)
        return result
