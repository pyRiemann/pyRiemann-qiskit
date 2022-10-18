import firebase_admin
import os
from firebase_admin import credentials, firestore
from .firebase_cert import certificate


class FirebaseConnector():
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
        except Exception:
            env_certificate = os.environ("FIREBASE_CERTIFICATE")
            cred = credentials.Certificate(env_certificate)
        cred = credentials.Certificate(certificate)
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
        return self._datasets
