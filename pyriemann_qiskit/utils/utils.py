from .mean import mean_functions
from .distance import distance_functions


def is_qfunction(string):
    """Indicates if the function is a mean or a distance introduced in this library.

    Return True is "string" represents a
    mean or a distance introduced in this library.

    Parameters
    ----------
    string: str
        A string representation of the mean/distance.

    Returns
    -------
    is_qfunction : boolean
        True if "string" represents a mean or a distance introduced in this library.

    Notes
    -----
    .. versionadded:: 0.2.0

    """
    return string[0] == "q" and (
        (string in mean_functions) or (string in distance_functions)
    )
