import pytest
from pyriemann_qiskit.utils.hyper_params_factory import (gen_zz_feature_map,
                                                         get_spsa)


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
    with pytest.raises(ValueError):
        feature_map.parameters


def test_get_spsa_default():
    """Test to create spsa with default parameters"""
    spsa = get_spsa()
    assert spsa._parameters[4] == 4.0
    assert spsa._maxiter == 40
    assert spsa._skip_calibration


def test_get_spsa_unknown_auto_calibration():
    """Test to create spsa with all none control parameters"""
    spsa = get_spsa(c=(None, None, None, None, None))
    for i in range(5):
        # Should use qiskit default values
        assert spsa._parameters[i] is not None
    assert not spsa._skip_calibration


def test_get_spsa_custom():
    """Test to create spsa with custom parameters"""
    spsa = get_spsa(max_trials=100, c=(0.0, 1.0, 2.0, 3.0, 4.0))
    for i in range(5):
        assert spsa._parameters[i] == i
    assert spsa._skip_calibration
    assert spsa._maxiter == 100
