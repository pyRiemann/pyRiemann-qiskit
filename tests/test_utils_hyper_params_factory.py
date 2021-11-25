import pytest
from pyriemann_qiskit.utils.hyper_params_factory import gen_zz_feature_map


@pytest.mark.parametrize(
    'entanglement', ['full', 'linear', 'circular', 'sca']
)
def test_gen_zz_feature_map_entangl_strings(entanglement):
    """Test gen_zz_feature_map with different string options of entanglement"""
    n_features = 2
    feature_map = gen_zz_feature_map(entanglement=entanglement)(n_features)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_idx(get_pauli_z_linear_entangl_idx):
    """Test gen_zz_feature_map with valid indices value"""
    n_features = 2
    reps = 2
    indices = get_pauli_z_linear_entangl_idx(reps, n_features)
    feature_map_handle = gen_zz_feature_map(reps=reps, entanglement=indices)
    feature_map = feature_map_handle(n_features)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_handle(get_pauli_z_linear_entangl_handle):
    """Test gen_zz_feature_map with a valid callable"""
    n_features = 2
    indices = get_pauli_z_linear_entangl_handle(n_features)
    feature_map = gen_zz_feature_map(entanglement=indices)(n_features)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_invalid_value():
    """Test gen_zz_feature_map with uncorrect value"""
    n_features = 2
    feature_map = gen_zz_feature_map(entanglement="invalid")(n_features)
    try:
        feature_map.parameters
        raise Exception("Invalid option should raise a ValueError exception")
    except ValueError:
        pass
