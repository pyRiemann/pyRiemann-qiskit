from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
import inspect


def gen_zz_feature_map(reps=2, entanglement='linear'):
    """Return a callable that generate a ZZFeatureMap.
    A feature map encodes data into a quantum state.
    A ZZFeatureMap is a second-order Pauli-Z evolution circuit.

    Parameters
    ----------
    reps : int (default 2)
        The number of repeated circuits, greater or equal to 1.
    entanglement : str | list[list[list[int]]] | \
                   Callable[int, list[list[list[int]]]]
        Specifies the entanglement structure.
        Entanglement structure can be provided with indices or string.
        Possible string values are: 'full', 'linear', 'circular' and 'sca'.
        Consult [1]_ for more details on entanglement structure.

    Returns
    -------
    ret : ZZFeatureMap
        An instance of ZZFeatureMap

    Raises
    ------
    ValueError
        Raised if ``reps`` is lower than 1.

    References
    ----------
    .. [1] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.NLocal.html
    """
    if reps < 1:
        raise ValueError("Parameter reps must be superior \
                          or equal to 1 (Got %d)" % reps)

    return lambda n_features: ZZFeatureMap(feature_dimension=n_features,
                                           reps=reps,
                                           entanglement=entanglement)


# Valid gates for two local circuits
gates = ['ch', 'cx', 'cy', 'cz', 'crx', 'cry', 'crz',
         'h', 'i', 'id', 'iden',
         'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz',
         's', 'sdg', 'swap',
         'x', 'y', 'z', 't', 'tdg']


def _check_gates_in_blocks(blocks):
    if isinstance(blocks, list):
        for gate in blocks:
            if gate not in gates:
                raise ValueError("Gate %s is not a valid gate" % gate)
    else:
        if blocks not in gates:
            raise ValueError("Gate %s is not a valid gate"
                             % blocks)


def gen_two_local(reps=3, rotation_blocks=['ry', 'rz'],
                  entanglement_blocks='cz'):
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
        An instance of a TwoLocal circuit

    Raises
    ------
    ValueError
        Raised if ``rotation_blocks`` or ``entanglement_blocks`` contain
        a non valid gate

    References
    ----------
    .. [1] \
        https://qiskit.org/documentation/stable/0.19/stubs/qiskit.circuit.library.TwoLocal.html
    """
    if reps < 1:
        raise ValueError("Parameter reps must be superior \
                          or equal to 1 (Got %d)" % reps)

    _check_gates_in_blocks(rotation_blocks)

    _check_gates_in_blocks(entanglement_blocks)

    return lambda n_features: TwoLocal(n_features,
                                       rotation_blocks,
                                       entanglement_blocks, reps=reps)


def get_spsa(max_trials=40, c=(None, None, None, None, 4.0)):
    """Return an instance of SPSA.
    SPSA [1, 2]_ is an algorithmic method for optimizing systems
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
        An instance of SPSA

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

    def calibrate(loss, initial_point=initial_point,
                  c=initial_pertubation,
                  stability_constant=stability_constant,
                  target_magnitude=None,
                  alpha=alpha, gamma=gamma,
                  modelspace=False, max_evals_grouped=1):
        return SPSA.calibrate(loss, initial_point, c,
                              stability_constant, target_magnitude,
                              alpha, gamma, modelspace, max_evals_grouped)

    spsa.calibrate = calibrate
    return spsa


def get_spsa_parameters(spsa):
    """Return the default values of the `calibrate` method of
    an SPSA instance. See [1]_ for implementation details.

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
    return (signature.parameters["initial_point"].default[0],
            signature.parameters["c"].default,
            signature.parameters["alpha"].default,
            signature.parameters["gamma"].default,
            signature.parameters["stability_constant"].default)
