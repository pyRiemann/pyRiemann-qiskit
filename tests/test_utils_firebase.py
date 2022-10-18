from pyriemann_qiskit.utils import FirebaseConnector

def test_firebase_connector():
    mock_data = {
        'dataset1' : {
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
    assert connector.datasets['dataset2']['subject2']['pipeline1']['true_labels'][0] == 1
    assert connector.datasets['dataset2']['subject2']['pipeline1']['predicted_labels'][0] == 0


