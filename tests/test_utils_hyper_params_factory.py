import pytest
from pyriemann_qiskit.utils.hyper_params_factory import gen_zz_feature_map


@pytest.mark.parametrize(
    'entanglement', ['full', 'linear', 'circular', 'sca']
)
def test_gen_zz_feature_map_entangl_strings(entanglement):
    """Test gen_zz_feature_map with different string options of entanglement"""
    feature_dim = 2
    feature_map = gen_zz_feature_map(entanglement=entanglement)(feature_dim)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_idx(get_pauli_z_linear_entangl_idx):
    """Test gen_zz_feature_map with valid indices value"""
    feature_dim = 2
    reps = 2
    indices = get_pauli_z_linear_entangl_idx(reps, feature_dim)
    feature_map_handle = gen_zz_feature_map(reps=reps, entanglement=indices)
    feature_map = feature_map_handle(feature_dim)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_handle(get_pauli_z_linear_entangl_handle):
    """Test gen_zz_feature_map with a valid callable"""
    feature_dim = 2
    indices = get_pauli_z_linear_entangl_handle(feature_dim)
    feature_map = gen_zz_feature_map(entanglement=indices)(feature_dim)
    assert isinstance(feature_map.parameters, set)


def test_gen_zz_feature_map_entangl_invalid_value():
    """Test gen_zz_feature_map with uncorrect value"""
    feature_dim = 2
    feature_map = gen_zz_feature_map(entanglement="invalid")(feature_dim)
    try:
        feature_map.parameters
        raise Exception("Invalid option should raise a ValueError exception")
    except ValueError:
        pass
