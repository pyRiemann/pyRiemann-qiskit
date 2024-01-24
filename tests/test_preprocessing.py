import numpy as np
from pyriemann_qiskit.utils.preprocessing import NdRobustScaler
from sklearn.preprocessing import RobustScaler


def test_ndrobustscaler(get_covmats):
    n_matrices, n_features = 5, 3

    X = get_covmats(n_matrices, n_features)

    scaler = NdRobustScaler()
    transformed_X = scaler.fit_transform(X)

    assert transformed_X.shape == X.shape

    # Check that each feature is scaled using RobustScaler
    for i in range(n_features):
        feature_before_scaling = X[:, i, :]
        feature_after_scaling = transformed_X[:, i, :]

        # Use RobustScaler to manually scale the feature and compare
        manual_scaler = RobustScaler()
        manual_scaler.fit(feature_before_scaling)
        manual_scaled_feature = manual_scaler.transform(feature_before_scaling)

        np.testing.assert_allclose(
            feature_after_scaling, manual_scaled_feature, rtol=1e-5
        )
