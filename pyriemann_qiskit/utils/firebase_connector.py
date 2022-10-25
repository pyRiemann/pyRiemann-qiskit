import firebase_admin
import os
from warnings import warn
try:
    from firebase_admin import credentials, firestore
except Exception:
    warn("""No firebase_admin found. Firebase connector \
         can only run with mock data.""")
from .firebase_cert import certificate


class FirebaseConnector():
    """
    A connector to Firebase.
    It gets/adds data to Firestore.
    (noSql database).

    Format of the data is as follows:

    ```
    datasets: {
        bi2012: {
            subject_01: {
                pipeline1: {
                    true_labels: [...],
                    predicted_labels: [...]
                }
            }
        }
    }
    ```

    Parameters
    ----------
    mock_data : Dict (default: None)
        If provided, it will skip the connection to firestore
        and use these data instead.
        It is useful for testing or
        for working with a local instance of a database.

    Notes
    -----
    .. versionadded:: 0.0.3
    """
    def __init__(self, mock_data=None) -> None:
        self._db = None
        self._datasets = {}
        self._collection = None
        self.mock_data = mock_data
        self._connect()
        self._read_stream()

    def _connect(self):
        if self.mock_data is not None:
            return
        cred = None
        try:
            cred = credentials.Certificate(certificate)
        except ValueError:
            env_certificate = eval(os.environ["FIREBASE_CERTIFICATE"])
            cred = credentials.Certificate(env_certificate)
        firebase_admin.initialize_app(cred)
        self._db = firestore.client()

    def _read_stream(self):
        if self.mock_data is not None:
            self._datasets = self.mock_data
            return
        self._collection = self._db.collection(u'datasets')
        stream = self._collection.stream()
        for dataset in stream:
            self._datasets[dataset.id] = dataset.to_dict()

    def add(self, dataset: str, subject: str, pipeline: str,
            true_labels: list, predicted_labels: list):
        """
        Add a data to firestore (or to the mock data if provided).

        Parameters
        ----------
        dataset : str
            The name of the dataset
        subject: str
            The name of the subject
        pipeline: str
            A string representation of the pipeline
        true_labels: list[str]
            The list of true labels provided to the pipeline
        predicted_labels: list[str]
            The list of predicted labels returned by the pipeline

        Raises
        ------
        KeyError
            Raised if there is already an entry for this
            combination of subject and pipeline.

        Notes
        -----
        .. versionadded:: 0.0.3
        """
        if dataset not in self.datasets:
            self._datasets[dataset] = {}
        dataset_dict = self._datasets[dataset]
        if subject not in dataset_dict:
            dataset_dict[subject] = {}
        subject_dict = dataset_dict[subject]
        if pipeline in subject_dict:
            raise KeyError(dataset +
                           '.' + subject +
                           '.' + pipeline +
                           " already exists.")
        subject_dict[pipeline] = {}
        pipeline_dict = subject_dict[pipeline]
        pipeline_dict["true_labels"] = true_labels
        pipeline_dict["predicted_labels"] = predicted_labels
        if not self.mock_data:
            self._collection.document(dataset).set(dataset_dict)

    @property
    def datasets(self):
        """
        Get the data from Firestore
        (or the mock data if provided).
        Data are readonly.

        Format of the data is as follows:

        Returns
        -------
        self : A representation of the database
        """
        return self._datasets


class Cache():
    """
    Layer of abstraction over FirebaseConnector.
    It facilitates adding/removing data
    for a particular dataset and pipeline.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.
    pipeline: Pipeline
        The sklearn pipeline use for the analysis of the dataset.
    mock_data: Dict (default: None)
        Mock data that can be passed to the FirebaseConnector.

    See Also
    --------
    FirebaseConnector

    Notes
    -----
    .. versionadded:: 0.0.3
    """
    def __init__(self, dataset_name: str, pipeline, mock_data=None) -> None:
        self._dataset_name = dataset_name
        self._pipeline = pipeline
        self._connector = FirebaseConnector(mock_data=mock_data)

    def _get_pipeline_dict(self, subject):
        key = str(self._pipeline)
        return self._connector.datasets[self._dataset_name][subject][key]

    def add(self, subject, true_labels, predicted_labels):
        """
        Add the prediction of the pipeline to the dataset.

        Parameters
        ----------
        subject: int|str
            The subject whom data were used for prediction.
        true_labels: List
            The true labels
        predicted_labels: The predicted labels

        See Also
        --------
        FirebaseConnector

        Notes
        -----
        .. versionadded:: 0.0.3
        """
        self._connector.add(self._dataset_name, subject, str(self._pipeline),
                            true_labels, predicted_labels)

    def get_result(self, subject):
        """
        Get a subject's predicted labels obtain
        with the pipeline pass as a parameter in the constructor.

        Parameters
        ----------
        subject: int|str
            A subject in the dataset.

        Return
        --------
        true_labels: List
            The true labels.
        predicted_labels: List
            The predicted labels obtained for this subject
            with the pipeline passed a a parameter in the constructor.

        Notes
        -----
        .. versionadded:: 0.0.3
        """
        results = self._get_pipeline_dict(subject)
        return results["true_labels"], results["predicted_labels"]

    def data(self):
        return self._connector.datasets
