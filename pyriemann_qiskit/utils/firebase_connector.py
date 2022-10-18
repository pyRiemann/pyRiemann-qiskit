import firebase_admin
from firebase_admin import credentials, firestore

cert = {
  "type": "service_account",
  "project_id": "pyriemann-qiskit-4d224",
  "private_key_id": "fc00a3d144ae933e56221dcc34b39eba3f642277",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC2HUA3+nLrPcDl\nxtZDNIe1VpKa71gXwqnmX9h0XZZcrTijT9QENDG4irpv3EyrpsIgJkdts0+/kZ67\ndcVqDVdNaivt/QPnmr3AJwG5YAvY/mIf8PLu4KZwbWkYtcVJUkhKIu4wdpg6fypX\nofaGwT9VVNmJcl8UFg6OerkVw6SOjS4kvHKKl7qlVzlZEdNaRgGf/ErHF6zIzpmd\njpQQ4CMS9fjXP8OH4nIR150NCt1uUU2pnvyfEb0JDiQcK/KvdjkREMr0UoNzYzLf\nzYiPN7wsxzg4qM90fEsOkkP63p1/q01PZVPiABcq5MlfCEWUKSno1I/z6+s+iJ6i\neyT4Rc6fAgMBAAECggEABLZ9EvXIQ441j7y1jsetO2QiJyBKh1LiUrPoRhql6YAS\nFyfZtMIlB0GP0ZVB5q9KFsrmzk/1oF+EXBWQJk5yE+7LdD03/KZkgrR/tuxlYhgz\nMeBqaQQnSoGKVrldgag/dquwvlBTYDAqSFpIpvQZT3wCpSvwifSIK+fBIm8NsyKp\nAYAi2KfF6lH3tsUdsaUszWbK6oMR9DcqlDBNkxZIkEwpuw9m0gUqTpDXWeQPv2dA\nU6+1oy4DgZwI8GJc4rA49EUSFcogEb5s1bzXsGuBuX2M1bRF8UeHh3kgVionY5Pa\nwPvbqOElCDAd5We2ViR8jpuizVenfhukYz7tq7GpAQKBgQDcaNt5ilKE2DKzzuNf\nDhUXAn2O+oI6kMhjbMW/ybhPZPTrvCzL37MGngn9p9/rtbXPLjW46MnmXx7jf4gx\n8pOlKv97wW+iLczjEbGpylvIXgfHjgwHRYuAhUcNj9GwkEsYHlMPhRoTBgGXeScM\nH57pNEbVEKjwbRqbtR667zuzVQKBgQDThV3mxLSFLvi+LuLpnd8Jn14U9fHA0Tay\nD77nFkV26asCWuc2PBynRDvS6EZ4cYNiQ9vIGynZIsIwPyj+xJEfS8y6PR5fevJM\nwvzcxGB1d+n1AWN6D+0L5PfrqEJtRwmEKNP7Mei/HN1yRaMTIzVRJNOOD5Hys/m1\nz+EbJh8iIwKBgGCV9q+BJP4f+7/9xneOJUHLDpiMQGtHrPic3t9Xl1P7arSu5naE\n1d/te3VTjnWtUDm1B0e7g1ZXkqKg7V7t7TNw7zodHz1TkrhitZcxHR1lz0Tsg2rg\nV+x++w7/WiVkZfPwvfdMHYv+ks39AlZ2uN160htmTJHnTcS0Dv5d/axtAoGAPFuJ\nMQmebYa6yiI88+bttyQ3x4lq49ePYP4nPm/XgJgrCTABXDOJBZ3t6EAJo+LYV9j/\nRTmjFmfZu8S6IQDcXG7Xy6kXGq3NqGPXnfOXhs2iABzWUwVqYgODT0ajNeWYbJLt\n8ncKcEZ/VlVStGpOk4oLqDT/M59xuRTJqmKVbbsCgYEAxWyW2J46WXkWgckYe6C/\ngbno+NEiSJD7W+wn9hWyOHnDRhpTmlKDHcv7Id/+hWCfvoQe0YuMZv7jt6VBwOrh\nXgHGzNvcmFPiZrh9pwN2li7ZkMfq6mlUW1R14UQurRqelhtTbXLfbVh1DtCg5Quy\n2MQIA2fkKGoja77EGkZNFGI=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-r9uba@pyriemann-qiskit-4d224.iam.gserviceaccount.com",
  "client_id": "110995456625994522554",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-r9uba%40pyriemann-qiskit-4d224.iam.gserviceaccount.com"
}

class FirebaseConnector():
    def __init__(self) -> None:
        self._db = None
        self._datasets = {}
        self._collection = None
        self._connect()
        self._read_stream()

    def _connect(self):
        cred = credentials.Certificate(cert)
        firebase_admin.initialize_app(cred)
        self._db = firestore.client()
    
    def _read_stream(self):
        self._collection = self._db.collection(u'datasets')
        stream = self._collection.stream()
        for dataset in stream:
            self._datasets[dataset.id] = dataset.to_dict()
    
    def add(self, dataset: str, subject: str, pipeline: str, true_labels: list, predicted_labels: list):
        if not dataset in self.datasets:
            self._datasets[dataset] = {}
        dataset_dict = self._datasets[dataset]; 
        if not subject in dataset_dict:
            dataset_dict[subject] = {}
        subject_dict = dataset_dict[subject]
        if pipeline in subject_dict:
            raise KeyError(dataset + '.' + subject + '.' + pipeline + " already exists.")
        else:
            subject_dict[pipeline] = {}
            pipeline_dict = subject_dict[pipeline]
            pipeline_dict["true_labels"] = true_labels
            pipeline_dict["predicted_labels"] = predicted_labels
        self._collection.document(dataset).set(dataset_dict)


    @property
    def datasets(self):
        return self._datasets
    

connector = FirebaseConnector()
print(connector.datasets)
connector.add('py.BI.EEG.2012-GIPSA', 'subject_01', 'TS+QuanticSVM2', [1, 0], [0, 1])


