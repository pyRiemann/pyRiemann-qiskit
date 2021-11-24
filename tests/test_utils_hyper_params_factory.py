import pytest
from pyriemann_qiskit.utils.hyper_params_factory import gen_zz_feature_map

@pytest.mark.parametrize(
    'entanglement', ['full', 'linear', 'circular', 'sca']
)
def test_gen_zz_feature_map_entanglement_string_values(entanglement):
    """Test gen_zz_feature_map with different values of entanglement"""
    feature_dim = 2
    feature_map = gen_zz_feature_map(entanglement=entanglement)(feature_dim)
    assert type(feature_map.parameters) == type(set())

def test_gen_zz_feature_map_entanglement_indices_value(get_zz_feature_map_linear_entanglement_indices):
    """Test gen_zz_feature_map with valid indices value"""
    feature_dim = 2
    reps=2
    indices = get_zz_feature_map_linear_entanglement_indices(reps, feature_dim)
    feature_map = gen_zz_feature_map(reps=reps, entanglement=indices)(feature_dim)
    assert type(feature_map.parameters) == type(set())

def test_gen_zz_feature_map_entanglement_callable_value(get_zz_feature_map_linear_entanglement_callable):
    """Test gen_zz_feature_map with valid indices value"""
    feature_dim = 2
    reps=2
    indices = get_zz_feature_map_linear_entanglement_callable(feature_dim)
    feature_map = gen_zz_feature_map(reps=reps, entanglement=indices)(feature_dim)
    assert type(feature_map.parameters) == type(set())

def test_gen_zz_feature_map_entanglement_invalid_value():
    """Test gen_zz_feature_map with uncorrect string value"""
    feature_dim = 2
    feature_map = gen_zz_feature_map(entanglement="invalid")(feature_dim)
    try:
        feature_map.parameters
        raise Exception("Option is invalid: it should raise a ValueError exception")
    except ValueError:
        pass