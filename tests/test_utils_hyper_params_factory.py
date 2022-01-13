import pytest
from pyriemann_qiskit.utils.hyper_params_factory import (gen_zz_feature_map,
                                                         gen_two_local, gates,
                                                         get_spsa)


class TestGenZZFeatureMap:

    @pytest.mark.parametrize(
        'entanglement', ['full', 'linear', 'circular', 'sca']
    )
    def test_entangl_strings(self, entanglement):
        """Test gen_zz_feature_map with different string options of entanglement"""
        n_features = 2
        feature_map = gen_zz_feature_map(entanglement=entanglement)(n_features)
        assert isinstance(feature_map.parameters, set)


    def test_entangl_idx(self, get_pauli_z_linear_entangl_idx):
        """Test gen_zz_feature_map with valid indices value"""
        n_features = 2
        reps = 2
        indices = get_pauli_z_linear_entangl_idx(reps, n_features)
        feature_map_handle = gen_zz_feature_map(reps=reps, entanglement=indices)
        feature_map = feature_map_handle(n_features)
        assert isinstance(feature_map.parameters, set)


    def test_entangl_handle(self, get_pauli_z_linear_entangl_handle):
        """Test gen_zz_feature_map with a valid callable"""
        n_features = 2
        indices = get_pauli_z_linear_entangl_handle(n_features)
        feature_map = gen_zz_feature_map(entanglement=indices)(n_features)
        assert isinstance(feature_map.parameters, set)


    def test_entangl_invalid_value(self):
        """Test gen_zz_feature_map with uncorrect value"""
        n_features = 2
        feature_map = gen_zz_feature_map(entanglement="invalid")(n_features)
        with pytest.raises(ValueError):
            feature_map.parameters


def test_gen_two_local_default():
    """Test default values of gen_zz_feature_map"""
    n_features = 2
    two_local_handle = gen_two_local()
    two_local = two_local_handle(n_features)
    assert two_local._num_qubits == n_features
    assert len(two_local._rotation_blocks) == 2
    assert len(two_local._entanglement_blocks) == 1


@pytest.mark.parametrize('rotation_blocks', gates)
@pytest.mark.parametrize('entanglement_blocks', gates)
def test_gen_two_local_strings(rotation_blocks, entanglement_blocks):
    """Test gen_two_local with different string options"""
    n_features = 2
    two_local_handle = gen_two_local(rotation_blocks=rotation_blocks,
                                     entanglement_blocks=entanglement_blocks)
    two_local = two_local_handle(n_features)
    assert isinstance(two_local._rotation_blocks, list)
    assert isinstance(two_local._entanglement_blocks, list)


def test_gen_two_local_list():
    """Test gen_two_local with a list as rotation and entanglement blocks"""
    n_features = 2
    rotation_blocks = ['cx', 'cz']
    entanglement_blocks = ['rx', 'rz']
    two_local_handle = gen_two_local(rotation_blocks=rotation_blocks,
                                     entanglement_blocks=entanglement_blocks)
    two_local = two_local_handle(n_features)
    assert isinstance(two_local._rotation_blocks, list)
    assert isinstance(two_local._entanglement_blocks, list)


def test_gen_two_local_invalid_string():
    """Test gen_two_local with invalid strings option"""
    rotation_blocks = 'invalid'
    entanglement_blocks = 'invalid'
    with pytest.raises(ValueError):
        gen_two_local(rotation_blocks=rotation_blocks,
                      entanglement_blocks=entanglement_blocks)


def test_gen_two_local_invalid_list():
    """Test gen_two_local with invalid strings option"""
    rotation_blocks = ['invalid', 'invalid']
    entanglement_blocks = ['invalid', 'invalid']
    with pytest.raises(ValueError):
        gen_two_local(rotation_blocks=rotation_blocks,
                      entanglement_blocks=entanglement_blocks)


def test_get_spsa_default():
    """Test to create spsa with default parameters"""
    spsa = get_spsa()
    assert spsa._parameters[4] == 4.0
    assert spsa._maxiter == 40
    assert spsa._skip_calibration


def test_get_spsa_auto_calibration():
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
