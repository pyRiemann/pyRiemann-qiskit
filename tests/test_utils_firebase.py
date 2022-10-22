from pyriemann_qiskit.utils import FirebaseConnector
from pyriemann_qiskit.classification import QuantumClassifierWithDefaultRiemannianPipeline
from pyriemann_qiskit.utils.firebase_connector import Cache
from sklearn.metrics import balanced_accuracy_score
import time

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
    def __init__(self):
        self.code = "MockDataset"
        self.subjects = [1, 2]
        self.data = {
            1: {
                "X": [100] * 10,
                "y": [0] * 10
            },
            2: {
                "X": [1000] * 10,
                "y": [1] * 10
            }
        }
    
    def get_data(self, subject):
        return self.data[subject]


def test_cache():
    dataset = MockDataset()
    pipeline = QuantumClassifierWithDefaultRiemannianPipeline(shots=None)
    cache = Cache(dataset, pipeline)
    scores = {}
    times = {}
    for subject in dataset.subjects:

        start = time.time()
        X, y = dataset.get_data(subject)
        pipeline.fit(X)
        y_pred = pipeline.predict(X)
        end = time.time()

        times[subject] = end - start
        cache.add(subject, y, y_pred)
        score = balanced_accuracy_score(y, y_pred)
        scores[subject] = score


    for subject in dataset.subjects:
        start = time.time()
        y, y_pred = cache.get_result(subject)
        end = time.time()
        score = balanced_accuracy_score(y, y_pred)
        assert score == scores[subject]
        assert times[subject] > end - start