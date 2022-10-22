from pyriemann_qiskit.utils import FirebaseConnector, Cache
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def test_firebase_connection():
    # Should retrieve correct certificate
    assert not FirebaseConnector() is None


def test_firebase_connector():
    mock_data = {
        'dataset1': {
            'subject1': {
                'pipeline1': {
                    'true_labels': [1, 0],
                    'predicted_labels': [0, 1]
                }
            }
        }
    }
    connector = FirebaseConnector(mock_data=mock_data)
    assert connector.datasets == mock_data
    connector.add('dataset2', 'subject2', 'pipeline1', [1], [0])
    pipeline_result = \
        connector.datasets['dataset2']['subject2']['pipeline1']
    assert pipeline_result['true_labels'][0] == 1
    assert pipeline_result['predicted_labels'][0] == 0


class MockDataset():
    def __init__(self, dataset_gen, n_subjects: int):
        self.code = "MockDataset"
        self.subjects = range(n_subjects)
        self.data = {}
        for subject in self.subjects:
            self.data[subject] = dataset_gen()

    def get_data(self, subject):
        return self.data[subject]


def test_cache(get_dataset):
    def dataset_gen():
        return get_dataset(n_samples=10, n_features=5,
                           n_classes=2, type="rand")

    dataset = MockDataset(dataset_gen, n_subjects=3)
    pipeline = make_pipeline(StandardScaler(), SVC(C=3.0))
    cache = Cache(dataset, pipeline, mock_data={})
    scores = {}

    for subject in dataset.subjects:
        X, y = dataset.get_data(subject)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        cache.add(subject, y, y_pred)
        score = balanced_accuracy_score(y, y_pred)
        scores[subject] = score

    for subject in dataset.subjects:
        y, y_pred = cache.get_result(subject)
        score = balanced_accuracy_score(y, y_pred)
        assert score == scores[subject]
