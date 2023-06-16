"""Module for pipelines."""
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.utils.hyper_params_factory import (
    gen_zz_feature_map,
    gen_two_local,
    get_spsa,
)
from pyriemann_qiskit.classification import QuanticVQC, QuanticSVM


class QuantumClassifierWithDefaultRiemannianPipeline(
    BaseEstimator, ClassifierMixin, TransformerMixin
):

    """Default pipeline with Riemann Geometry and a quantum classifier.

    Projects the data into the tangent space of the Riemannian manifold
    and applies quantum classification.

    The type of quantum classification (quantum SVM or VQC) depends on
    the value of the parameters.

    Data are entangled using a ZZFeatureMap. A SPSA optimizer and a two-local
    circuits are used in addition when the VQC is selected.



    Parameters
    ----------
    nfilter : int (default: 1)
        The number of filter for the xDawnFilter.
        The number of components selected is 2 x nfilter.
    dim_red : TransformerMixin (default: PCA())
        A transformer that will reduce the dimension of the feature,
        after the data are projected into the tangent space.
    gamma : float | None (default:None)
        Used as input for sklearn rbf_kernel which is used internally.
        See [1]_ for more information about gamma.
    C : float (default: 1.0)
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        Note, if pegasos is enabled you may want to consider
        larger values of C.
    max_iter: int | None (default: None)
        number of steps in Pegasos or SVC.
        If None, respective default values for Pegasos and (Q)SVC
        are used. The default value for Pegasos is 1000.
        For (Q)SVC it is -1 (that is not limit).
    shots : int | None (default: 1024)
        Number of repetitions of each circuit, for sampling.
        If None, classical computation will be performed.
    feature_entanglement : str | list[list[list[int]]] | \
                   Callable[int, list[list[list[int]]]]
        Specifies the entanglement structure for the ZZFeatureMap.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular' and 'sca'.
        Consult [2]_ for more details on entanglement structure.
    feature_reps : int (default: 2)
        The number of repeated circuits for the ZZFeatureMap,
        greater or equal to 1.
    spsa_trials : int (default: 40)
        Maximum number of iterations to perform using SPSA optimizer.
    two_local_reps : int (default: 3)
        The number of repetition for the two-local cricuit.
    params: Dict (default: {})
        Additional parameters to pass to the nested instance
        of the quantum classifier.
        See QuanticClassifierBase, QuanticVQC and QuanticSVM for
        a complete list of the parameters.

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    XdawnCovariances
    TangentSpace
    gen_zz_feature_map
    gen_two_local
    get_spsa
    QuanticVQC
    QuanticSVM
    QuanticClassifierBase

    References
    ----------
    .. [1] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html

    .. [2] \
        https://qiskit.org/documentation/stable/0.36/stubs/qiskit.circuit.library.NLocal.html

    """

    def __init__(
        self,
        nfilter=1,
        dim_red=PCA(),
        gamma="scale",
        C=1.0,
        max_iter=None,
        shots=1024,
        feature_entanglement="full",
        feature_reps=2,
        spsa_trials=None,
        two_local_reps=None,
        params={},
    ):
        self.nfilter = nfilter
        self.dim_red = dim_red
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.shots = shots
        self.feature_entanglement = feature_entanglement
        self.feature_reps = feature_reps
        self.spsa_trials = spsa_trials
        self.two_local_reps = two_local_reps
        self.params = params

        is_vqc = spsa_trials and two_local_reps
        is_quantum = shots is not None

        feature_map = gen_zz_feature_map(feature_reps, feature_entanglement)
        # verbose is passed as an additional parameter to quantum classifiers.
        self.verbose = "verbose" in params and params["verbose"]
        if is_vqc:
            self._log("QuanticVQC chosen.")
            clf = QuanticVQC(
                optimizer=get_spsa(spsa_trials),
                gen_var_form=gen_two_local(two_local_reps),
                gen_feature_map=feature_map,
                shots=self.shots,
                quantum=is_quantum,
                **params
            )
        else:
            self._log("QuanticSVM chosen.")
            clf = QuanticSVM(
                quantum=is_quantum,
                gamma=gamma,
                C=C,
                max_iter=max_iter,
                gen_feature_map=feature_map,
                shots=shots,
                **params
            )

        self._pipe = make_pipeline(
            XdawnCovariances(nfilter=nfilter), TangentSpace(), dim_red, clf
        )

    def _log(self, trace):
        if self.verbose:
            print("[QuantumClassifierWithDefaultRiemannianPipeline] ", trace)

    def fit(self, X, y):
        """Train the riemann quantum classifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : QuantumClassifierWithDefaultRiemannianPipeline instance
            The QuantumClassifierWithDefaultRiemannianPipeline instance
        """

        self.classes_ = np.unique(y)
        self._pipe.fit(X, y)
        return self

    def score(self, X, y):
        """Return the accuracy.
        You might want to use a different metric by using sklearn
        cross_val_score

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.
        y : ndarray, shape (n_trials,)
            Predicted target vector relative to X.

        Returns
        -------
        accuracy : double
            Accuracy of predictions from X with respect y.
        """
        return self._pipe.score(X, y)

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            Class labels for samples in X.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """Return the probabilities associated with predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, 0] == True if the nth sample is assigned to 1st class;
            prob[n, 1] == True if the nth sample is assigned to 2nd class.
        """

        return self._pipe.predict_proba(X)

    def transform(self, X):
        """Transform the data into feature vectors.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            ndarray of trials.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the data.
            the dimension of the feature vector depends on
            `n_filter` and `dim_red`.
        """
        return self._pipe.transform(X)
