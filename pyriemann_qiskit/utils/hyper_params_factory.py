import inspect

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qiskit.circuit.library import (
    ZZFeatureMap,
    ZFeatureMap,
    PauliFeatureMap,
    TwoLocal,
)
from qiskit_algorithms.optimizers import SPSA


def gen_x_feature_map(reps=2):
    """Return a callable that generates a XFeatureMap.

    A feature map encodes data into a quantum state.
    A XFeatureMap is a first-order Pauli-X evolution circuit (no entanglement).
    See [1]_ for more details.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.

    Returns
    -------
    ret : XFeatureMap
        An instance of XFeatureMap.

    Raises
    ------
    ValueError
        Raised if ``reps`` is lower than 1.

    References
    ----------
    .. [1] \
        https://docs.quantum.ibm.com/api/qiskit/0.44/qiskit.circuit.library.PauliFeatureMap

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    if reps < 1:
        raise ValueError(f"Parameter reps must be superior or equal to 1 (Got {reps})")

    return lambda n_features: PauliFeatureMap(
        feature_dimension=n_features,
        paulis=["X"],
        reps=reps,
        data_map_func=None,
        parameter_prefix="x",
        insert_barriers=False,
        name="XFeatureMap",
    )


def gen_z_feature_map(reps=2):
    """Return a callable that generates a ZFeatureMap.

    A feature map encodes data into a quantum state.
    A ZFeatureMap is a first-order Pauli-Z evolution circuit (no entanglement).
    See [1]_ for more details.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.

    Returns
    -------
    ret : ZFeatureMap
        An instance of ZFeatureMap.

    Raises
    ------
    ValueError
        Raised if ``reps`` is lower than 1.

    References
    ----------
    .. [1] \
        https://docs.quantum.ibm.com/api/qiskit/0.44/qiskit.circuit.library.ZFeatureMap

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    if reps < 1:
        raise ValueError(f"Parameter reps must be superior or equal to 1 (Got {reps})")

    return lambda n_features: ZFeatureMap(feature_dimension=n_features, reps=reps)


def gen_zz_feature_map(reps=2, entanglement="linear"):
    """Return a callable that generates a ZZFeatureMap.

    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.
    See [1]_ for more details.


    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.
    entanglement : str | list[list[list[int]]] | \
                   Callable[int, list[list[list[int]]]]
        Specifies the entanglement structure.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular' and 'sca'.
        See [2]_ for more details on entanglement structure.

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap.

    Raises
    ------
    ValueError
        Raised if ``reps`` is lower than 1.

    References
    ----------
    .. [1] \
        https://docs.quantum.ibm.com/api/qiskit/0.44/qiskit.circuit.library.ZZFeatureMap
    .. [2] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.NLocal.html

    Notes
    -----
    .. versionadded:: 0.0.1
    """
    if reps < 1:
        raise ValueError(f"Parameter reps must be superior or equal to 1 (Got {reps})")

    return lambda n_features: ZZFeatureMap(
        feature_dimension=n_features, reps=reps, entanglement=entanglement
    )


# Valid gates for two local circuits
gates = [
    "ch",
    "cx",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "h",
    "id",
    "rx",
    "rxx",
    "ry",
    "ryy",
    "rz",
    "rzx",
    "rzz",
    "s",
    "sdg",
    "swap",
    "x",
    "y",
    "z",
    "t",
    "tdg",
]


def _check_gates_in_blocks(blocks):
    if isinstance(blocks, list):
        for gate in blocks:
            if gate not in gates:
                raise ValueError("Gate %s is not a valid gate" % gate)
    else:
        if blocks not in gates:
            raise ValueError("Gate %s is not a valid gate" % blocks)


def gen_two_local(reps=3, rotation_blocks=["ry", "rz"], entanglement_blocks="cz"):
    """Return a callable that generate a TwoLocal circuit.

    The two-local circuit is a parameterized circuit consisting
    of alternating rotation layers and entanglement layers [1]_.

    Parameters
    ----------
    reps : int (default 3)
        Specifies how often a block consisting of a rotation layer
        and entanglement layer is repeated.
    rotation_blocks : str | list[str]
        The gates used in the rotation layer.
        Valid string values are defined in `gates`.
    entanglement_blocks : str | list[str]
        The gates used in the entanglement layer.
        Valid string values are defined in `gates`.

    Returns
    -------
    ret : TwoLocal
        An instance of a TwoLocal circuit.

    Raises
    ------
    ValueError
        Raised if ``rotation_blocks`` or ``entanglement_blocks`` contain
        a non valid gate.

    References
    ----------
    .. [1] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.TwoLocal.html
    """
    if reps < 1:
        raise ValueError(
            "Parameter reps must be superior \
                          or equal to 1 (Got %d)"
            % reps
        )

    _check_gates_in_blocks(rotation_blocks)

    _check_gates_in_blocks(entanglement_blocks)

    return lambda n_features: TwoLocal(
        n_features, rotation_blocks, entanglement_blocks, reps=reps
    )


def get_spsa(max_trials=40, c=(None, None, None, None, 4.0)):
    """Return an instance of SPSA.

    SPSA [1]_, [2]_ is an algorithmic method for optimizing systems
    with multiple unknown parameters.
    For more details, see [3]_ and [4]_.

    Parameters
    ----------
    max_trials : int (default:40)
        Maximum number of iterations to perform.
    c : tuple[float | None] (default:(None, None, None, None, 4.0))
        The 5 control parameters for SPSA algorithms, namely:
        the initial point, the intial perturbation, alpha, gamma
        and the stability constant.
        See [3]_ for implementation details. This function set the
        default value of the control parameters for the `calibrate` method of
        the implementation.

    Returns
    -------
    ret : SPSA
        An instance of SPSA.

    References
    ----------
    .. [1] Spall, J. C. (2012), “Stochastic Optimization,”
           in Handbook of Computational Statistics:
           Concepts and Methods (2nd ed.)
           (J. Gentle, W. Härdle, and Y. Mori, eds.),
           Springer−Verlag, Heidelberg, Chapter 7, pp. 173–201.
           dx.doi.org/10.1007/978-3-642-21551-3_7

    .. [2] Spall, J. C. (1999), "Stochastic Optimization:
           Stochastic Approximation and Simulated Annealing,"
           in Encyclopedia of Electrical and Electronics Engineering
           (J. G. Webster, ed.),
           Wiley, New York, vol. 20, pp. 529–542

    .. [3] \
        https://qiskit.org/documentation/stable/0.36/stubs/qiskit.algorithms.optimizers.SPSA.html

    .. [4] https://www.jhuapl.edu/SPSA/#Overview
    """
    spsa = SPSA(maxiter=max_trials)
    initial_point = [c[0]] if c[0] else [0]
    initial_pertubation = c[1] if c[1] else 0.2
    alpha = c[2] if c[2] else 0.602
    gamma = c[3] if c[3] else 0.101
    stability_constant = c[4] if c[4] else 0

    def calibrate(
        loss,
        initial_point=initial_point,
        c=initial_pertubation,
        stability_constant=stability_constant,
        target_magnitude=None,
        alpha=alpha,
        gamma=gamma,
        modelspace=False,
        max_evals_grouped=1,
    ):
        return SPSA.calibrate(
            loss,
            initial_point,
            c,
            stability_constant,
            target_magnitude,
            alpha,
            gamma,
            modelspace,
            max_evals_grouped,
        )

    spsa.calibrate = calibrate
    return spsa


def get_spsa_parameters(spsa):
    """Return the default values of the `calibrate` method of an SPSA instance.

    See [1]_ for implementation details.

    Parameters
    ----------
    spsa : SPSA
        The SPSA instance.

    Returns
    -------
    params : The default values of the control parameters for
        the calibrate method in this order:
        initial point, initial perturbation, alpha,
        gamma and stability constant.

    Notes
    -----
    .. versionadded:: 0.0.2

    References
    ----------
    .. [1] \
        https://qiskit.org/documentation/stable/0.36/stubs/qiskit.algorithms.optimizers.SPSA.calibrate.html
    """
    signature = inspect.signature(spsa.calibrate)
    return (
        signature.parameters["initial_point"].default[0],
        signature.parameters["c"].default,
        signature.parameters["alpha"].default,
        signature.parameters["gamma"].default,
        signature.parameters["stability_constant"].default,
    )


def create_mixer_qiskit_default(_angle):
    r"""Return the default mixing operator with QAOA.
    (qiskit implementation)

    .. math::
        H_X = \sum_{i}^{N} X_i

    See [1]_ for details.

    Parameters
    ----------
    angle : float
        The angle of the gates' rotation.
        Not used. Just kept for compatibility with other
        `create_mixer`

    Returns
    -------
    mixer : Callable[[int, boolean], QuantumCircuit]
        A method that returns None,
        forcing qiskit to use its own implementation of the mixer.

    Notes
    -----
    .. versionadded:: 0.4.0

    References
    ----------
    .. [1] \
        https://dice.cyfronet.pl/papers/JPlewa_JSienko_msc_v2.pdf
    """

    def mixer_qiskit_default(n_qubits, use_params=False):
        return None

    return mixer_qiskit_default


def create_mixer_rotational_X_gates(angle):
    r"""Return the default mixing operator with QAOA.

    .. math::
        H_X = \sum_{i}^{N} X_i

    See [1]_ for details.

    Parameters
    ----------
    angle : float
        The angle of the gates' rotation.

    Returns
    -------
    mixer : Callable[[int, boolean], QuantumCircuit]
        A method that create the mixer with the correct angle.
        This method takes into parameters the number of qubits in the circuit,
        and a parameter `use_params`.
        if True, `use_params` create a Parameter in the circuit (same for all gates)

    Notes
    -----
    .. versionadded:: 0.4.0

    References
    ----------
    .. [1] \
        https://dice.cyfronet.pl/papers/JPlewa_JSienko_msc_v2.pdf
    """

    def mixer_X(n_qubits, use_params=False):
        qr = QuantumRegister(n_qubits)
        mixer = QuantumCircuit(qr)
        p = Parameter("beta")
        for qr_ in qr:
            if use_params:
                mixer.rx(p + angle, qr_)
            else:
                mixer.rx(angle, qr_)
        return mixer

    return mixer_X


def create_mixer_rotational_XY_gates(angle):
    r"""Return the XY mixer.

    .. math::
        H_{XY} = \sum_{i}^{N-1} \left( X_i X_{i+1} + Y_i Y_{i+1} \right),

    See [1]_ for details.

    Parameters
    ----------
    angle : float
        The angle of the gates' rotation.

    Returns
    -------
    mixer : Callable[[int, boolean], QuantumCircuit]
        A method that create the mixer with the correct angle.
        This method takes into parameters the number of qubits in the circuit,
        and a parameter `use_params`.
        if True, `use_params` create a Parameter in the circuit (same for all gates)

    Notes
    -----
    .. versionadded:: 0.4.0

    References
    ----------
    .. [1] \
        https://dice.cyfronet.pl/papers/JPlewa_JSienko_msc_v2.pdf
    """

    def mixer_XY(n_qubits, use_params):
        qr = QuantumRegister(n_qubits)
        mixer = QuantumCircuit(qr)
        p = Parameter("beta")
        for i in range(n_qubits - 1):
            if use_params:
                mixer.rxx(p + angle, qr[i], qr[i + 1])
            else:
                mixer.rxx(angle, qr[i], qr[i + 1])
            mixer.ryy(0 + angle, qr[i], qr[i + 1])
        return mixer

    return mixer_XY


def create_mixer_rotational_XZ_gates(angle):
    r"""Return a mixing operator with XZ gates.

    .. math::
        H_{\text{mix}} = \sum_{i}^{N-1} \left( Z_{i-1} X_i - X_i Z_{i+1} \right).

    See [1]_ for details.

    Parameters
    ----------
    angle : float
        The angle of the gates' rotation.

    Returns
    -------
    mixer : Callable[[int, boolean], QuantumCircuit]
        A method that create the mixer with the correct angle.
        This method takes into parameters the number of qubits in the circuit,
        and a parameter `use_params`.
        if True, `use_params` create a Parameter in the circuit (same for all gates)

    Notes
    -----
    .. versionadded:: 0.4.0

    References
    ----------
    .. [1] \
        https://dice.cyfronet.pl/papers/JPlewa_JSienko_msc_v2.pdf
    """

    def mixer_XZ(n_qubits, use_params):
        qr = QuantumRegister(n_qubits)
        mixer = QuantumCircuit(qr)

        p = Parameter("beta")
        for i in range(n_qubits - 1):
            if use_params:
                mixer.rzx(p + angle, qr[i], qr[i + 1])
            else:
                mixer.rzx(angle, qr[i], qr[i + 1])
            mixer.rzx(0 + angle, qr[i], qr[i + 1]).inverse(annotated=True)
        return mixer

    return mixer_XZ
