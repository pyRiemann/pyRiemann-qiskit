import firebase_admin
import os
# import ast
from firebase_admin import credentials, firestore
from .firebase_cert import certificate


class FirebaseConnector():
    """
    A connector to Firebase.
    It gets/adds data to Firestore
    (noSql database)

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
        if self.mock_data:
            return
        cred = None
        try:
            cred = credentials.Certificate(certificate)
        except ValueError:
            env_certificate = \
                eval(os.environ["FIREBASE_CERTIFICATE"])
            cred = credentials.Certificate(env_certificate)
        firebase_admin.initialize_app(cred)
        self._db = firestore.client()

    def _read_stream(self):
        if self.mock_data:
            self._datasets = self.mock_data
            return
        self._collection = self._db.collection(u'datasets')
        stream = self._collection.stream()
        for dataset in stream:
            self._datasets[dataset.id] = dataset.to_dict()

    def add(self, dataset: str, subject: str, pipeline: str,
            true_labels: list, predicted_labels: list):
        """
        Add a data to firestore (or to the mock data if provided)

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
        else:
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


def Cache():

    def __init__(self, dataset, pipeline):
        self._dataset = dataset
        self._pipeline = pipeline
        self._connector = FirebaseConnector()

    def _get_pipeline_dict(self):
        key = str(self._pipeline)
        return self._connector.datasets[self._dataset.code][key]

    def add(self, subject, true_labels, predicted_labels):
        self._connector.add(self._dataset.code, str(self._pipeline),
                            subject, true_labels, predicted_labels)

    def get_result(self, subject):
        return self._get_pipeline_dict()[subject]
