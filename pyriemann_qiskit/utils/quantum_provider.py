"""Module containing helpers for IBM quantum backends providers and simulators"""

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
