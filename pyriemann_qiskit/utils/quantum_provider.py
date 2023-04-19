"""Module containing helpers for IBM quantum backends
   providers and simulators."""

from qiskit_ibm_provider import IBMProvider


def get_provider():
    """Return an IBM quantum provider.

    Returns
    -------
    provider : IBMProvider
        An instance of IBMProvider.

    Notes
    -----
    .. versionadded:: 0.0.4
    """
    return IBMProvider.get_provider(hub='ibm-q')


def get_devices(provider, min_qubits):
    """Returns all real remote quantum backends,
    available with the account token and having at least
    `min_qubits`

    Parameters
    ----------
    provider: IBMProvider
        An instance of IBMProvider.
    min_qubits: int
        The minimun of qubits.

    Returns
    -------
    devices: IBMQBackend[]
        A list of compatible backends.

    Notes
    -----
    .. versionadded:: 0.0.4
    """
    def filters(device):
        return (
            device.configuration().n_qubits >= min_qubits
            and not device.configuration().simulator
            and device.status().operational)
    devices = provider.backends(filters=filters)
    return devices
